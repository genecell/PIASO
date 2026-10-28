//! Leiden community detection (Traag, Waltman & van Eck 2019) for modularity
//! with a resolution parameter: parallel, and deterministic.
//!
//! **One graph and one seed give one partition.** The number of threads, the
//! number of cores of the machine, the scheduling of the threads and the
//! layout of the graph in memory do not enter the result. Three rules make
//! that so, one for each way a parallel program usually loses it:
//!
//! 1. *Order of moves.* Local moving runs in rounds. Every decision of a
//!    round reads the state the round began with; the moves that are kept
//!    are applied together, behind a barrier.
//! 2. *Random numbers.* A random number is the hash of (seed, level, round,
//!    node): it is the same whichever thread asks for it, and whenever.
//! 3. *Sums.* Edge weights are integers (fixed point, 30 bits of the largest
//!    weight of the graph), so community totals and aggregated weights are
//!    exact, and an exact sum does not depend on the order of its terms.
//!    Gains are then computed in floating point from identical integers.
//!
//! **Identity.** A node is named by the smallest original index among the
//! cells it holds, not by its position in memory: hashes, the order of
//! refinement and exact ties all use that name. The reordering of the nodes
//! along the graph is therefore a matter of speed alone, and can be changed
//! without changing a label.
//!
//! **The rule of a round.** In one round a community may lose nodes or gain
//! nodes, not both. Each proposed move has a priority (its hash); where a
//! community has both leavers and joiners the side holding the best priority
//! wins and the other side waits for the next round. Two nodes swapping into
//! each other's community -- what makes plain synchronous moving oscillate --
//! needs a community that receives while it loses, which the rule forbids.
//! Many nodes may still join one community in the same round, which is what
//! lets large communities form in few rounds. The move of best priority is
//! always kept, so every round with a proposal makes progress.
//!
//! **Quality function.** Q = 1/(2m) * sum_c [ W_c - gamma * K_c^2 / (2m) ],
//! with W_c the weight of edges inside c (both directions) and K_c the summed
//! strength of its nodes. Moving node v from A to B changes it by
//! (k_vB - k_vA\v) - gamma * k_v * (K_B - K_A\v) / (2m), up to the 1/m factor.
//!
//! **Levels.** Local moving, refinement inside each community (communities in
//! parallel, each refined by one thread), aggregation of the refined
//! communities into nodes whose initial partition is the unrefined one;
//! repeated until no community holds more than one node. Self weight of an
//! aggregate node never enters a move, only its strength, so it is not stored.

use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering::Relaxed};

/// Fractional bits of the fixed-point weights: the largest weight of the
/// graph is 2^30, so a 32-bit weight has room and 1e-9 of it is resolved.
const UNIT_BITS: u32 = 30;
/// Below this many items a step runs on the calling thread: handing a few
/// thousand nodes to the pool costs more than visiting them.
const SMALL: usize = 4096;
/// Rounds of one local move. Never reached in any measurement (the largest
/// seen is a few hundred); a guard, reported to the caller when it is hit.
const MAX_ROUNDS: usize = 1000;
/// Graphs with fewer nodes than this run on one thread.
const SINGLE: usize = 1 << 15;
/// Communities per block of the aggregation.
const AGG_BLOCK: usize = 1024;
const INTERRUPTED: &str = "interrupted";
const ALONE: u32 = u32::MAX;
const NOPRIO: u64 = u64::MAX;

/// What the algorithm needs from a graph: its size, node strengths and
/// identities, and the weighted neighbours of a node (never the node itself).
trait Adj: Sync {
    fn n(&self) -> usize;
    fn strength(&self) -> &[u64];
    fn ident(&self) -> &[u32];
    fn degree(&self, v: usize) -> usize;
    fn for_each_edge<F: FnMut(u32, u64)>(&self, v: usize, f: F);
}

/// The first level: the caller's graph with 32-bit integer weights, the
/// diagonal removed (it only counts towards strength).
struct Fine {
    indptr: Vec<usize>,
    indices: Vec<u32>,
    weights: Vec<u32>,
    strength: Vec<u64>,
    ident: Vec<u32>,
}

impl Adj for Fine {
    fn n(&self) -> usize {
        self.strength.len()
    }
    fn strength(&self) -> &[u64] {
        &self.strength
    }
    fn ident(&self) -> &[u32] {
        &self.ident
    }
    fn degree(&self, v: usize) -> usize {
        self.indptr[v + 1] - self.indptr[v]
    }
    #[inline]
    fn for_each_edge<F: FnMut(u32, u64)>(&self, v: usize, mut f: F) {
        for e in self.indptr[v]..self.indptr[v + 1] {
            f(self.indices[e], self.weights[e] as u64);
        }
    }
}

/// An aggregated level: 64-bit weights, since they are sums.
struct Coarse {
    indptr: Vec<usize>,
    indices: Vec<u32>,
    weights: Vec<u64>,
    strength: Vec<u64>,
    ident: Vec<u32>,
}

impl Adj for Coarse {
    fn n(&self) -> usize {
        self.strength.len()
    }
    fn strength(&self) -> &[u64] {
        &self.strength
    }
    fn ident(&self) -> &[u32] {
        &self.ident
    }
    fn degree(&self, v: usize) -> usize {
        self.indptr[v + 1] - self.indptr[v]
    }
    #[inline]
    fn for_each_edge<F: FnMut(u32, u64)>(&self, v: usize, mut f: F) {
        for e in self.indptr[v]..self.indptr[v + 1] {
            f(self.indices[e], self.weights[e]);
        }
    }
}

/// Renumber labels to 0..k in order of first appearance.
fn renumber(labels: &mut [u32]) -> usize {
    let n = labels.len();
    let mut map = vec![u32::MAX; n.max(1) + 1];
    let mut next = 0u32;
    for l in labels.iter_mut() {
        let li = *l as usize;
        if li >= map.len() {
            map.resize(li + 1, u32::MAX);
        }
        if map[li] == u32::MAX {
            map[li] = next;
            next += 1;
        }
        *l = map[li];
    }
    next as usize
}

#[inline]
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// The random number of (seed, context, round, identity): a counter-based
/// generator, so it does not matter which thread asks or in what order.
#[inline]
fn hash4(seed: u64, ctx: u64, round: u64, id: u64) -> u64 {
    mix(mix(mix(seed ^ 0x9E3779B97F4A7C15).wrapping_add(ctx)).wrapping_add(round.wrapping_mul(0xD1B54A32D192ED03))
        ^ id.wrapping_mul(0x9E3779B97F4A7C15))
}

/// What every level needs to know.
struct Params<'a> {
    gamma: f64,
    /// gamma / 2m, with 2m in the integer unit of the weights.
    scale: f64,
    /// The seed of every hash. Priorities always use one; `random` says
    /// whether refinement draws its merges or takes the best.
    seed: u64,
    random: bool,
    /// Randomness of the refinement (Leiden's theta; igraph's beta).
    theta: f64,
    /// 1 / (weight unit): refinement's gains are in units of the largest
    /// weight, the scale theta is meant for.
    unit: f64,
    profile: bool,
    /// Set from outside when the caller wants the run to stop (Ctrl-C).
    cancel: &'a AtomicBool,
}

/// Counters of a run, for the profile and for the round guard.
#[derive(Default)]
struct Tally {
    rounds: usize,
    capped: bool,
    t_move: f64,
    t_refine: f64,
    t_aggregate: f64,
}

/// A proposed move.
#[derive(Clone, Copy)]
struct Proposal {
    node: u32,
    /// The community to join, or `ALONE`.
    target: u32,
    /// Priority: the smaller wins. Unique, since it ends in the identity.
    key: u64,
    /// What the move is worth over staying, from the state the round began
    /// with.
    margin: f64,
}

/// The best community for `v`, from the state as it stands. `None` when it
/// stays where it is.
#[inline]
#[allow(clippy::too_many_arguments)]
fn propose<G: Adj>(g: &G, v: u32, comm: &[AtomicU32], k_comm: &[AtomicU64], size: &[AtomicU32], p: &Params,
                   ctx: u64, round: u64, buf: &mut Vec<(u32, u64, u32)>) -> Option<Proposal> {
    let vi = v as usize;
    let ident = g.ident();
    let strength = g.strength();
    let a = comm[vi].load(Relaxed);
    let kv = strength[vi] as f64;
    buf.clear();
    g.for_each_edge(vi, |u, w| buf.push((comm[u as usize].load(Relaxed), w, ident[u as usize])));
    // weights per community: a node has few neighbours, so sorting them is
    // cheaper than a table as large as the graph for every thread
    buf.sort_unstable_by_key(|x| x.0);
    let k_a = (k_comm[a as usize].load(Relaxed) - strength[vi]) as f64;
    let mut w_a = 0u64;
    let mut best = a;
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_id = u32::MAX;
    let mut i = 0;
    while i < buf.len() {
        let c = buf[i].0;
        let mut w = 0u64;
        // a community is named by the neighbour of smallest identity in it
        let mut cid = u32::MAX;
        while i < buf.len() && buf[i].0 == c {
            w += buf[i].1;
            cid = cid.min(buf[i].2);
            i += 1;
        }
        if c == a {
            w_a = w;
            continue;
        }
        let gain = w as f64 - p.scale * kv * k_comm[c as usize].load(Relaxed) as f64;
        if gain > best_gain || (gain == best_gain && cid < best_id) {
            best = c;
            best_gain = gain;
            best_id = cid;
        }
    }
    let stay = w_a as f64 - p.scale * kv * k_a;
    // alone is worth 0; a node that is alone already has nothing to gain by it
    let (target, gain) = if best_gain > stay && best_gain >= 0.0 {
        (best, best_gain)
    } else if stay < 0.0 && size[a as usize].load(Relaxed) > 1 {
        (ALONE, 0.0)
    } else {
        return None;
    };
    let id = ident[vi];
    let key = (hash4(p.seed, ctx, round, id as u64) & 0xFFFF_FFFF_0000_0000) | id as u64;
    Some(Proposal { node: v, target, key, margin: gain - stay })
}

/// For every proposal, the summed strength of the proposals of better
/// priority in the same group (`group` names it; `u32::MAX` is no group).
/// `crowd` is scratch, one counter per community, zero on entry and on exit.
fn ahead(props: &[Proposal], group: &(dyn Fn(&Proposal) -> u32 + Sync), strength: &[u64], crowd: &[AtomicU32]) -> Vec<AtomicU64> {
    let small = props.len() < SMALL;
    let out: Vec<AtomicU64> = (0..props.len()).map(|_| AtomicU64::new(0)).collect();
    // a proposal alone in its group has nothing ahead of it: most are
    let count = |x: &Proposal| {
        let c = group(x);
        if c != u32::MAX {
            crowd[c as usize].fetch_add(1, Relaxed);
        }
    };
    let shared = |(i, x): (usize, &Proposal)| -> Option<(u32, u64, u32)> {
        let c = group(x);
        if c != u32::MAX && crowd[c as usize].load(Relaxed) > 1 { Some((c, x.key, i as u32)) } else { None }
    };
    let clear = |x: &Proposal| {
        let c = group(x);
        if c != u32::MAX {
            crowd[c as usize].store(0, Relaxed);
        }
    };
    let sums = |members: &[(u32, u64, u32)]| {
        let mut sum = 0u64;
        for &(_, _key, i) in members {
            out[i as usize].store(sum, Relaxed);
            sum += strength[props[i as usize].node as usize];
        }
    };
    if small {
        props.iter().for_each(count);
        let mut order: Vec<(u32, u64, u32)> = props.iter().enumerate().filter_map(shared).collect();
        props.iter().for_each(clear);
        order.sort_unstable();
        order.chunk_by(|a, b| a.0 == b.0).for_each(sums);
    } else {
        props.par_iter().for_each(count);
        let mut order: Vec<(u32, u64, u32)> = props.par_iter().enumerate().filter_map(shared).collect();
        props.par_iter().for_each(clear);
        order.par_sort_unstable();
        order.par_chunk_by(|a, b| a.0 == b.0).for_each(sums);
    }
    out
}

/// Local moving in synchronous rounds. `comm` is modified in place; returns
/// whether any node moved.
///
/// A round keeps a proposed move when two things hold.
///
/// *A community loses nodes or gains nodes, not both.* Where a community has
/// leavers and joiners, the side holding the best priority wins.
///
/// *The move still pays when every proposal ahead of it is made too.* Nodes
/// that join one community in the same round each reckoned without the
/// others; so did nodes that leave one. A move is therefore tested against
/// the community as it would be if all the proposals of better priority with
/// the same target had joined it, and all those of better priority from the
/// same community had left.
///
/// Together they make every kept move worth at least what was reckoned for
/// it: the members of the community it joins all stay, nobody joins the one
/// it leaves, and the totals it was tested against are no smaller than the
/// real ones. So each round with a proposal raises the modularity -- the
/// proposal of best priority is always kept -- and the rounds must end.
fn local_move<G: Adj>(g: &G, comm: &mut [u32], p: &Params, ctx: u64, tally: &mut Tally) -> bool {
    let n = g.n();
    let t_a = std::time::Instant::now();
    let strength = g.strength();
    // SAFETY: AtomicU32 has the size and alignment of u32, and `comm` is
    // borrowed mutably for the whole call, so nothing else reads it.
    let comm: &[AtomicU32] = unsafe { &*(comm as *mut [u32] as *const [AtomicU32]) };
    let k_comm: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
    let size: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
    for v in 0..n {
        let c = comm[v].load(Relaxed) as usize;
        k_comm[c].fetch_add(strength[v], Relaxed);
        size[c].fetch_add(1, Relaxed);
    }
    // best priority among the nodes leaving / joining each community
    let emit: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(NOPRIO)).collect();
    let acc: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(NOPRIO)).collect();
    let listed: Vec<AtomicU8> = (0..n).map(|_| AtomicU8::new(0)).collect();
    let crowd: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
    let mut active: Vec<u32> = (0..n as u32).collect();
    let mut empties: Vec<u32> = Vec::new();
    let mut moved_any = false;
    let mut round = 0usize;
    let mut spent = [0.0f64; 5];
    let (mut n_active, mut n_props, mut n_moved) = (0usize, 0usize, 0usize);
    let t_b = std::time::Instant::now();
    while !active.is_empty() {
        if p.cancel.load(Relaxed) {
            break;
        }
        if round >= MAX_ROUNDS {
            tally.capped = true;
            break;
        }
        let r = round as u64;
        let t_1 = std::time::Instant::now();
        // 1. proposals, from the state as it is now: read-only, no locks
        let mut props: Vec<Proposal> = if active.len() < SMALL {
            let mut buf = Vec::new();
            active.iter().filter_map(|&v| propose(g, v, comm, &k_comm, &size, p, ctx, r, &mut buf)).collect()
        } else {
            active.par_iter()
                .map_init(Vec::new, |buf, &v| propose(g, v, comm, &k_comm, &size, p, ctx, r, buf))
                .filter_map(|x| x)
                .collect()
        };
        let t_2 = std::time::Instant::now();
        n_props += props.len();
        let small = props.len() < SMALL;
        let source = |x: &Proposal| comm[x.node as usize].load(Relaxed);
        // 2a. a community loses nodes or gains nodes in one round, not both
        let reserve = |x: &Proposal| {
            emit[source(x) as usize].fetch_min(x.key, Relaxed);
            if x.target != ALONE {
                acc[x.target as usize].fetch_min(x.key, Relaxed);
            }
        };
        let allowed = |x: &Proposal| -> bool {
            let a = source(x) as usize;
            // leaving a: no joiner of a may hold a better priority;
            // joining b: no leaver of b may
            emit[a].load(Relaxed) <= acc[a].load(Relaxed)
                && (x.target == ALONE || acc[x.target as usize].load(Relaxed) <= emit[x.target as usize].load(Relaxed))
        };
        let release = |x: &Proposal| {
            emit[source(x) as usize].store(NOPRIO, Relaxed);
            if x.target != ALONE {
                acc[x.target as usize].store(NOPRIO, Relaxed);
            }
        };
        let mut waiting: Vec<u32>;
        if small {
            props.iter().for_each(reserve);
            let ok: Vec<bool> = props.iter().map(allowed).collect();
            props.iter().for_each(release);
            waiting = props.iter().zip(&ok).filter(|x| !*x.1).map(|x| x.0.node).collect();
            let mut i = 0;
            props.retain(|_| {
                i += 1;
                ok[i - 1]
            });
        } else {
            props.par_iter().for_each(reserve);
            let ok: Vec<bool> = props.par_iter().map(allowed).collect();
            props.par_iter().for_each(release);
            waiting = props.par_iter().zip(&ok).filter(|x| !*x.1).map(|x| x.0.node).collect();
            props = props.par_iter().zip(&ok).filter(|x| *x.1).map(|x| *x.0).collect();
        }
        let t_3 = std::time::Instant::now();
        // 2b. the move still pays when the proposals ahead of it are made too
        let joined = ahead(&props, &|x| x.target, strength, &crowd); // ALONE is u32::MAX: no group
        let left = ahead(&props, &source, strength, &crowd);
        let pays = |(i, x): (usize, &Proposal)| -> bool {
            let kv = strength[x.node as usize] as f64;
            x.margin > p.scale * kv * (joined[i].load(Relaxed) + left[i].load(Relaxed)) as f64
        };
        let kept: Vec<bool> = if small {
            props.iter().enumerate().map(pays).collect()
        } else {
            props.par_iter().enumerate().map(pays).collect()
        };
        waiting.extend(props.iter().zip(&kept).filter(|x| !*x.1).map(|x| x.0.node));
        let t_4 = std::time::Instant::now();
        // 3. apply the kept moves together: exact integer totals, so the
        // order in which the threads add does not matter
        let apply = |(x, &k): (&Proposal, &bool)| {
            if k && x.target != ALONE {
                let vi = x.node as usize;
                let a = comm[vi].load(Relaxed) as usize;
                let kv = strength[vi];
                k_comm[a].fetch_sub(kv, Relaxed);
                size[a].fetch_sub(1, Relaxed);
                k_comm[x.target as usize].fetch_add(kv, Relaxed);
                size[x.target as usize].fetch_add(1, Relaxed);
                comm[vi].store(x.target, Relaxed);
            }
        };
        if small {
            props.iter().zip(kept.iter()).for_each(apply);
        } else {
            props.par_iter().zip(kept.par_iter()).for_each(apply);
        }
        // nodes that go alone: few, taken one by one. Which empty community
        // each gets names it internally and nothing more.
        let mut moved: Vec<(u32, u32)> = Vec::new();
        for (x, _) in props.iter().zip(&kept).filter(|x| *x.1 && x.0.target == ALONE) {
            let vi = x.node as usize;
            let a = comm[vi].load(Relaxed) as usize;
            loop {
                if empties.is_empty() {
                    empties = (0..n as u32).rev().filter(|&c| size[c as usize].load(Relaxed) == 0).collect();
                    if empties.is_empty() {
                        break;
                    }
                }
                let c = empties.pop().unwrap();
                if size[c as usize].load(Relaxed) == 0 {
                    let kv = strength[vi];
                    k_comm[a].fetch_sub(kv, Relaxed);
                    size[a].fetch_sub(1, Relaxed);
                    k_comm[c as usize].fetch_add(kv, Relaxed);
                    size[c as usize].fetch_add(1, Relaxed);
                    comm[vi].store(c, Relaxed);
                    moved.push((x.node, c));
                    break;
                }
            }
        }
        moved.extend(props.iter().zip(&kept).filter(|x| *x.1 && x.0.target != ALONE).map(|x| (x.0.node, x.0.target)));
        let t_5 = std::time::Instant::now();
        // 4. the next active set: the moves that had to wait, and the
        // neighbours of a moved node that are not in its new community. Read
        // after every move is applied, so it does not depend on their order.
        let mut next: Vec<u32> = Vec::new();
        for &v in &waiting {
            if listed[v as usize].swap(1, Relaxed) == 0 {
                next.push(v);
            }
        }
        let wake = |mut out: Vec<u32>, &(v, b): &(u32, u32)| {
            g.for_each_edge(v as usize, |u, _w| {
                if comm[u as usize].load(Relaxed) != b && listed[u as usize].swap(1, Relaxed) == 0 {
                    out.push(u);
                }
            });
            out
        };
        if moved.len() < SMALL {
            next = moved.iter().fold(next, wake);
        } else {
            let mut woken = moved.par_iter().fold(Vec::new, wake).reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            });
            next.append(&mut woken);
        }
        // in node order: the next round then walks memory front to back
        if next.len() < SMALL {
            next.sort_unstable();
            next.iter().for_each(|&v| listed[v as usize].store(0, Relaxed));
        } else {
            next.par_sort_unstable();
            next.par_iter().for_each(|&v| listed[v as usize].store(0, Relaxed));
        }
        if !moved.is_empty() {
            moved_any = true;
        }
        n_active += active.len();
        n_moved += moved.len();
        let t_6 = std::time::Instant::now();
        for (k, d) in [t_2 - t_1, t_3 - t_2, t_4 - t_3, t_5 - t_4, t_6 - t_5].iter().enumerate() {
            spent[k] += d.as_secs_f64();
        }
        active = next;
        round += 1;
    }
    tally.rounds += round;
    if p.profile && n > 100_000 {
        eprintln!("  move {} [{:.2} GB]: before {:.3}s, rounds {:.3}s = propose {:.2} one-way {:.2} ahead {:.2} apply {:.2} next {:.2}; per node: visited {:.2}, proposed {:.2}, moved {:.2}",
                  n, resident_gb(), (t_b - t_a).as_secs_f64(), t_b.elapsed().as_secs_f64(), spent[0], spent[1], spent[2], spent[3], spent[4],
                  n_active as f64 / n as f64, n_props as f64 / n as f64, n_moved as f64 / n as f64);
    }
    moved_any
}

/// exp, the same bits on every machine: the system's own may differ in the
/// last one, which is enough to change a random draw.
#[inline]
fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// Refinement: inside each community, singletons merge into a
/// well-connected sub-community, drawn with probability proportional to
/// exp(gain / theta) (or the best one, without randomness). Communities are
/// independent, so they are refined in parallel, each by one thread.
fn refine<G: Adj>(g: &G, comm: &[u32], n_comm: usize, p: &Params, ctx: u64) -> Vec<u32> {
    let n = g.n();
    let t_a = std::time::Instant::now();
    let strength = g.strength();
    let ident = g.ident();
    let mut k_comm = vec![0u64; n_comm];
    let mut start = vec![0usize; n_comm + 1];
    for v in 0..n {
        k_comm[comm[v] as usize] += strength[v];
        start[comm[v] as usize + 1] += 1;
    }
    for c in 0..n_comm {
        start[c + 1] += start[c];
    }
    let mut visit = vec![0u32; n];
    {
        let mut fill = start.clone();
        for v in 0..n {
            let c = comm[v] as usize;
            visit[fill[c]] = v as u32;
            fill[c] += 1;
        }
    }
    let mut parts: Vec<(u32, &mut [u32])> = Vec::with_capacity(n_comm);
    {
        let mut rest: &mut [u32] = &mut visit;
        for c in 0..n_comm {
            let (head, tail) = rest.split_at_mut(start[c + 1] - start[c]);
            if head.len() > 1 {
                parts.push((c as u32, head));
            }
            rest = tail;
        }
    }
    // Each entry is written by the one thread that refines the node's
    // community; atomics only so that the threads may share the arrays.
    let refined: Vec<AtomicU32> = (0..n as u32).map(AtomicU32::new).collect();
    let k_ref: Vec<AtomicU64> = strength.iter().map(|&x| AtomicU64::new(x)).collect();
    let size_ref: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(1)).collect();
    // weight from each node, then from each sub-community, to the rest of
    // its community
    let k_in: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
    let ext: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
    let sub = p.seed ^ 0x5EF1_4E00_0000_0000;
    type Scratch = (Vec<(u32, u64)>, Vec<(u32, f64, u64)>, Vec<(u64, u32, u32)>);
    let one = |(buf, cands, keys): &mut Scratch, (c, members): &mut (u32, &mut [u32])| {
        let c = *c;
        let kc = k_comm[c as usize] as f64;
        for &v in members.iter() {
            let mut sum = 0u64;
            g.for_each_edge(v as usize, |u, w| {
                if comm[u as usize] == c {
                    sum += w;
                }
            });
            k_in[v as usize].store(sum, Relaxed);
            ext[v as usize].store(sum, Relaxed);
        }
        // the members in an order drawn by their hashes (Leiden asks for a
        // random order); they stay together, close in memory
        keys.clear();
        keys.extend(members.iter().map(|&v| (hash4(sub, ctx, 0, ident[v as usize] as u64), ident[v as usize], v)));
        keys.sort_unstable();
        for &(_, _, vv) in keys.iter() {
            let v = vv as usize;
            let own = refined[v].load(Relaxed);
            if size_ref[own as usize].load(Relaxed) != 1 {
                continue; // already joined by another node
            }
            let kv = strength[v] as f64;
            let kin = k_in[v].load(Relaxed);
            if (kin as f64) < p.scale * kv * (kc - kv) {
                continue; // not well connected to its community
            }
            buf.clear();
            g.for_each_edge(v, |u, w| {
                let u = u as usize;
                if comm[u] == c {
                    buf.push((refined[u].load(Relaxed), w));
                }
            });
            buf.sort_unstable_by_key(|x| x.0);
            cands.clear();
            let mut best_gain = 0.0f64;
            let mut i = 0;
            while i < buf.len() {
                let t = buf[i].0;
                let mut w = 0u64;
                while i < buf.len() && buf[i].0 == t {
                    w += buf[i].1;
                    i += 1;
                }
                if t == own {
                    continue;
                }
                let kt = k_ref[t as usize].load(Relaxed) as f64;
                if (ext[t as usize].load(Relaxed) as f64) < p.scale * kt * (kc - kt) {
                    continue; // the sub-community is not well connected
                }
                let gain = (w as f64 - p.scale * kv * kt) * p.unit;
                if gain >= 0.0 {
                    cands.push((t, gain, w));
                }
                if gain > best_gain {
                    best_gain = gain;
                }
            }
            if cands.is_empty() {
                continue;
            }
            // a sub-community is named by the node that founded it
            if cands.len() > 1 {
                cands.sort_unstable_by_key(|x| ident[x.0 as usize]);
            }
            let mut pick: Option<(u32, u64)> = None;
            if p.random {
                // among the merges that do not lower the quality (staying
                // alone included), choose with probability proportional to
                // exp(gain / theta)
                let top = best_gain.max(0.0);
                // each weight once; one that is below 1e-17 of the best is
                // nothing, and costs no exp
                let weight = |gain: f64| -> f64 {
                    let d = (gain - top) / p.theta;
                    if d < -40.0 { 0.0 } else { exp(d) }
                };
                let alone = weight(0.0);
                let mut total = alone;
                for c in cands.iter_mut() {
                    c.1 = weight(c.1);
                    total += c.1;
                }
                let mut x = (hash4(sub, ctx, 1, ident[v] as u64) >> 11) as f64 / (1u64 << 53) as f64 * total;
                x -= alone;
                if x >= 0.0 {
                    for &(t, e, w) in cands.iter() {
                        if e > 0.0 {
                            x -= e;
                            pick = Some((t, w));
                            if x < 0.0 {
                                break;
                            }
                        }
                    }
                }
            } else {
                // the best merge; of equal ones, the first by name
                let mut g_best = 0.0f64;
                for &(t, gn, w) in cands.iter() {
                    if gn > g_best {
                        g_best = gn;
                        pick = Some((t, w));
                    }
                }
            }
            if let Some((t, w_vt)) = pick {
                let t = t as usize;
                ext[t].store(ext[t].load(Relaxed) + kin - 2 * w_vt, Relaxed);
                k_ref[t].fetch_add(strength[v], Relaxed);
                size_ref[t].fetch_add(1, Relaxed);
                size_ref[own as usize].store(0, Relaxed);
                refined[v].store(t as u32, Relaxed);
            }
        }
    };
    let t_b = std::time::Instant::now();
    if n < SMALL {
        let mut scratch: Scratch = (Vec::new(), Vec::new(), Vec::new());
        parts.iter_mut().for_each(|part| one(&mut scratch, part));
    } else {
        parts.par_iter_mut().for_each_init(|| (Vec::new(), Vec::new(), Vec::new()), one);
    }
    let t_c = std::time::Instant::now();
    let out = refined.into_iter().map(|x| x.into_inner()).collect();
    if p.profile && n > 100_000 {
        eprintln!("  refine {} [{:.2} GB]: before {:.3}s, communities {:.3}s, after {:.3}s", n, resident_gb(), (t_b - t_a).as_secs_f64(),
                  (t_c - t_b).as_secs_f64(), t_c.elapsed().as_secs_f64());
    }
    out
}

/// The graph of `labels` (0..k): node weights summed, edges between different
/// labels summed, neighbours sorted.
///
/// Rows are built in parallel, a bounded number of blocks of communities at
/// a time. A row has at most as many entries as its members have edges, so
/// every block has a place of known size in one scratch buffer, which is
/// allocated once and used again for the next blocks; what the rows turn out
/// to hold is then copied, in parallel, to its place in the result.
fn aggregate<G: Adj>(g: &G, labels: &[u32], k: usize) -> Coarse {
    use std::mem::MaybeUninit;
    let n = g.n();
    let t_a = std::time::Instant::now();
    let gs = g.strength();
    let gi = g.ident();
    let mut strength = vec![0u64; k];
    let mut ident = vec![u32::MAX; k];
    let mut count = vec![0usize; k + 1];
    // most entries a row can have: the edges of its members
    let mut most = vec![0usize; k];
    for v in 0..n {
        let l = labels[v] as usize;
        strength[l] += gs[v];
        ident[l] = ident[l].min(gi[v]);
        count[l + 1] += 1;
        most[l] += g.degree(v);
    }
    for i in 0..k {
        count[i + 1] += count[i];
    }
    let mut members = vec![0u32; n];
    {
        let mut fill = count.clone();
        for v in 0..n {
            let l = labels[v] as usize;
            members[fill[l]] = v as u32;
            fill[l] += 1;
        }
    }
    // One row into `ix`/`wt` from position `at`; returns its length.
    let row = |buf: &mut Vec<(u32, u64)>, a: usize, ix: &mut [MaybeUninit<u32>], wt: &mut [MaybeUninit<u64>], at: usize| -> usize {
        buf.clear();
        for &v in &members[count[a]..count[a + 1]] {
            g.for_each_edge(v as usize, |u, w| {
                let t = labels[u as usize];
                if t as usize != a {
                    buf.push((t, w));
                }
            });
        }
        buf.sort_unstable_by_key(|x| x.0);
        let (mut i, mut len) = (0, 0);
        while i < buf.len() {
            let t = buf[i].0;
            let mut w = 0u64;
            while i < buf.len() && buf[i].0 == t {
                w += buf[i].1;
                i += 1;
            }
            ix[at + len].write(t);
            wt[at + len].write(w);
            len += 1;
        }
        len
    };
    let n_blocks = k.div_ceil(AGG_BLOCK);
    let block_most: Vec<usize> = (0..n_blocks)
        .map(|b| most[b * AGG_BLOCK..((b + 1) * AGG_BLOCK).min(k)].iter().sum())
        .collect();
    let at_once = if k < SMALL { n_blocks.max(1) } else { 4 * rayon::current_num_threads().max(1) };
    let scratch: usize = block_most.chunks(at_once).map(|c| c.iter().sum::<usize>()).max().unwrap_or(0);
    let mut s_ix: Vec<u32> = Vec::with_capacity(scratch);
    let mut s_wt: Vec<u64> = Vec::with_capacity(scratch);
    let mut row_len = vec![0u32; k];
    // Reserved for the largest the result can be, not touched: only what is
    // written becomes memory in use, and the rest is given back at the end.
    let cap: usize = most.iter().sum();
    let mut indices: Vec<u32> = Vec::with_capacity(cap);
    let mut weights: Vec<u64> = Vec::with_capacity(cap);
    let (mut t_rows, mut t_copy) = (0.0f64, 0.0f64);
    let mut b0 = 0;
    while b0 < n_blocks {
        let b1 = (b0 + at_once).min(n_blocks);
        let t_r = std::time::Instant::now();
        // every block: its place in the scratch, its row lengths
        let mut jobs = Vec::with_capacity(b1 - b0);
        {
            let mut ix_r = s_ix.spare_capacity_mut();
            let mut wt_r = s_wt.spare_capacity_mut();
            let mut len_r = &mut row_len[b0 * AGG_BLOCK..(b1 * AGG_BLOCK).min(k)];
            for b in b0..b1 {
                let (lo, hi) = (b * AGG_BLOCK, ((b + 1) * AGG_BLOCK).min(k));
                let (ix, ix_rest) = ix_r.split_at_mut(block_most[b]);
                let (wt, wt_rest) = wt_r.split_at_mut(block_most[b]);
                let (ln, ln_rest) = len_r.split_at_mut(hi - lo);
                jobs.push((lo, ix, wt, ln, 0usize));
                ix_r = ix_rest;
                wt_r = wt_rest;
                len_r = ln_rest;
            }
        }
        let fill = |buf: &mut Vec<(u32, u64)>, (lo, ix, wt, ln, used): &mut (usize, &mut [MaybeUninit<u32>], &mut [MaybeUninit<u64>], &mut [u32], usize)| {
            for (i, l) in ln.iter_mut().enumerate() {
                let len = row(buf, *lo + i, ix, wt, *used);
                *l = len as u32;
                *used += len;
            }
        };
        if k < SMALL {
            let mut buf = Vec::new();
            jobs.iter_mut().for_each(|j| fill(&mut buf, j));
        } else {
            jobs.par_iter_mut().for_each_init(Vec::new, fill);
        }
        t_rows += t_r.elapsed().as_secs_f64();
        // what the rows hold, to its place in the result
        let t_c = std::time::Instant::now();
        let added: usize = jobs.iter().map(|j| j.4).sum();
        {
            let mut ix_r = &mut indices.spare_capacity_mut()[..added];
            let mut wt_r = &mut weights.spare_capacity_mut()[..added];
            let mut copies = Vec::with_capacity(jobs.len());
            for (_, ix, wt, _, used) in jobs.iter() {
                let (a, ix_rest) = ix_r.split_at_mut(*used);
                let (c, wt_rest) = wt_r.split_at_mut(*used);
                copies.push((&ix[..*used], &wt[..*used], a, c));
                ix_r = ix_rest;
                wt_r = wt_rest;
            }
            let copy = |(from_ix, from_wt, to_ix, to_wt): &mut (&[MaybeUninit<u32>], &[MaybeUninit<u64>], &mut [MaybeUninit<u32>], &mut [MaybeUninit<u64>])| {
                to_ix.copy_from_slice(from_ix);
                to_wt.copy_from_slice(from_wt);
            };
            if k < SMALL {
                copies.iter_mut().for_each(copy);
            } else {
                copies.par_iter_mut().for_each(copy);
            }
        }
        // SAFETY: the `added` entries past the length were copied just above
        // from entries that `row` wrote, and the capacity holds them.
        unsafe {
            indices.set_len(indices.len() + added);
            weights.set_len(weights.len() + added);
        }
        t_copy += t_c.elapsed().as_secs_f64();
        b0 = b1;
    }
    drop(s_ix);
    drop(s_wt);
    let mut indptr = Vec::with_capacity(k + 1);
    indptr.push(0usize);
    let mut at = 0usize;
    for &l in &row_len {
        at += l as usize;
        indptr.push(at);
    }
    indices.shrink_to_fit();
    weights.shrink_to_fit();
    if std::env::var_os("PIASO_LEIDEN_PROFILE").is_some() && n > 100_000 {
        eprintln!("  aggregate {} [{:.2} GB]: in all {:.3}s, rows {:.3}s, copy {:.3}s", n, resident_gb(), t_a.elapsed().as_secs_f64(), t_rows, t_copy);
    }
    Coarse { indptr, indices, weights, strength, ident }
}

/// Return freed memory to the system. The many buffers of a level are freed
/// into the allocator's own lists, where they would stay, counted against the
/// process, while the next level allocates again: 0.2 GB on 2.3 million nodes.
fn give_back() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: malloc_trim only releases memory the allocator holds free.
    unsafe {
        libc::malloc_trim(0);
    }
}

/// Memory in use by the process, for the profile.
fn resident_gb() -> f64 {
    std::fs::read_to_string("/proc/self/statm").ok()
        .and_then(|t| t.split_whitespace().nth(1).and_then(|x| x.parse::<f64>().ok()))
        .map(|pages| pages * 4096.0 / 1e9)
        .unwrap_or(0.0)
}

/// One level: local moving, refinement, aggregation. Returns the aggregate
/// graph, the map from this level's nodes to its nodes, and its initial
/// partition; or ``None`` when every community is already one node.
fn level<G: Adj>(g: &G, comm: &mut Vec<u32>, p: &Params, ctx: u64, tally: &mut Tally) -> Option<(Coarse, Vec<u32>, Vec<u32>)> {
    let t0 = std::time::Instant::now();
    let rounds_before = tally.rounds;
    local_move(g, comm, p, ctx, tally);
    let t_move = t0.elapsed().as_secs_f64();
    tally.t_move += t_move;
    let large = g.n() > 100_000;
    if large {
        give_back();
    }
    let n_comm = renumber(comm);
    if n_comm == g.n() {
        return None; // every community is one node: nothing left to merge
    }
    let t1 = std::time::Instant::now();
    let mut refined = refine(g, comm, n_comm, p, ctx);
    let t_refine = t1.elapsed().as_secs_f64();
    tally.t_refine += t_refine;
    if large {
        give_back();
    }
    let mut n_ref = renumber(&mut refined);
    if n_ref == g.n() {
        // refinement merged nothing: aggregate on the communities themselves,
        // so the next level is smaller
        refined = comm.clone();
        n_ref = n_comm;
    }
    // the next level's initial partition: each refined community's community
    let mut next_comm = vec![0u32; n_ref];
    for v in 0..g.n() {
        next_comm[refined[v] as usize] = comm[v];
    }
    let t2 = std::time::Instant::now();
    let agg = aggregate(g, &refined, n_ref);
    let t_agg = t2.elapsed().as_secs_f64();
    tally.t_aggregate += t_agg;
    if p.profile {
        eprintln!("leiden level [{:.2} GB]: {} nodes -> {} communities, {} refined; move {:.2}s ({} rounds) refine {:.2}s aggregate {:.2}s",
                  resident_gb(),
                  g.n(), n_comm, n_ref, t_move, tally.rounds - rounds_before, t_refine, t_agg);
    }
    Some((agg, refined, next_comm))
}

/// One run of Leiden from `initial` (or singletons). Returns the membership of
/// every node of `g0`, renumbered 0..k.
fn leiden_once(g0: &Fine, initial: Option<&[u32]>, p: &Params, max_levels: usize, iteration: u64, tally: &mut Tally) -> Vec<u32> {
    let n0 = g0.n();
    let mut to_node: Vec<u32> = (0..n0 as u32).collect(); // node of g0 -> node of the current level
    let mut comm: Vec<u32> = match initial {
        Some(p) => p.to_vec(),
        None => (0..n0 as u32).collect(),
    };
    let mut current: Option<Coarse> = None;
    for lv in 0..max_levels as u64 {
        if p.cancel.load(Relaxed) {
            break;
        }
        let ctx = iteration * 4096 + lv;
        let step = match &current {
            None => level(g0, &mut comm, p, ctx, tally),
            Some(g) => level(g, &mut comm, p, ctx, tally),
        };
        let Some((agg, refined, next_comm)) = step else { break };
        if n0 < SMALL {
            to_node.iter_mut().for_each(|x| *x = refined[*x as usize]);
        } else {
            to_node.par_iter_mut().for_each(|x| *x = refined[*x as usize]);
        }
        comm = next_comm;
        current = Some(agg);
    }
    let mut out: Vec<u32> = to_node.iter().map(|&x| comm[x as usize]).collect();
    renumber(&mut out);
    out
}

/// Modularity of `labels` on `g`, at resolution `gamma`; `self_w` is the
/// weight of each node's own loop.
fn modularity<G: Adj>(g: &G, labels: &[u32], gamma: f64, self_w: &[u64]) -> f64 {
    let k = labels.iter().map(|&l| l as usize + 1).max().unwrap_or(0);
    let mut inside = vec![0u64; k];
    let mut tot = vec![0u64; k];
    let strength = g.strength();
    for v in 0..g.n() {
        let c = labels[v] as usize;
        tot[c] += strength[v];
        inside[c] += self_w[v];
        g.for_each_edge(v, |u, w| {
            if labels[u as usize] as usize == c {
                inside[c] += w;
            }
        });
    }
    let two_m = strength.iter().sum::<u64>() as f64;
    if !(two_m > 0.0) {
        return 0.0;
    }
    let mut q = 0.0;
    for c in 0..k {
        q += inside[c] as f64 - gamma * (tot[c] as f64) * (tot[c] as f64) / two_m;
    }
    q / two_m
}

/// Relabel so the largest community is 0, ties by smallest node index: the
/// order a reader expects, and independent of internal numbering.
fn order_by_size(labels: &mut [u32]) {
    let k = labels.iter().map(|&l| l as usize + 1).max().unwrap_or(0);
    let mut size = vec![0usize; k];
    let mut first = vec![usize::MAX; k];
    for (v, &l) in labels.iter().enumerate() {
        size[l as usize] += 1;
        if first[l as usize] == usize::MAX {
            first[l as usize] = v;
        }
    }
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| size[b].cmp(&size[a]).then(first[a].cmp(&first[b])));
    let mut rank = vec![0u32; k];
    for (r, &c) in order.iter().enumerate() {
        rank[c] = r as u32;
    }
    for l in labels.iter_mut() {
        *l = rank[*l as usize];
    }
}

/// Edge weights of the caller's matrix, read in place.
#[derive(Clone, Copy)]
enum Weights<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
}

impl Weights<'_> {
    #[inline]
    fn get(&self, e: usize) -> f64 {
        match self {
            Weights::F32(d) => d[e] as f64,
            Weights::F64(d) => d[e],
        }
    }
    fn len(&self) -> usize {
        match self {
            Weights::F32(d) => d.len(),
            Weights::F64(d) => d.len(),
        }
    }
}

/// Check the caller's CSR arrays; returns the largest weight.
fn check(indptr: &[i64], indices: &[i32], data: Weights, n: usize) -> Result<f64, String> {
    if indptr.len() != n + 1 {
        return Err(format!("indptr has {} entries for {} nodes", indptr.len(), n));
    }
    if data.len() != indices.len() {
        return Err("indices and data differ in length".into());
    }
    if n > (u32::MAX - 2) as usize {
        return Err(format!("{} nodes are more than this implementation holds", n));
    }
    let mut last = 0i64;
    for &x in indptr {
        if x < last || x as usize > indices.len() {
            return Err("indptr is not a valid CSR index".into());
        }
        last = x;
    }
    if indptr[0] != 0 {
        return Err("indptr is not a valid CSR index".into());
    }
    let bad_col = indices.par_iter().find_any(|&&u| u < 0 || u as usize >= n);
    if let Some(u) = bad_col {
        return Err(format!("column index {} out of range", u));
    }
    let ok = |w: f64| w >= 0.0 && w.is_finite();
    let (bad, wmax) = (0..data.len()).into_par_iter()
        .fold(|| (None, 0.0f64), |(bad, m), e| {
            let w = data.get(e);
            if ok(w) { (bad, m.max(w)) } else { (bad.or(Some(w)), m) }
        })
        .reduce(|| (None, 0.0f64), |a, b| (a.0.or(b.0), a.1.max(b.1)));
    if let Some(w) = bad {
        return Err(format!("edge weight {} is negative or not finite", w));
    }
    Ok(wmax)
}

/// Reverse Cuthill-McKee order: breadth-first from a low-degree node of each
/// connected component, neighbours taken in increasing degree, then reversed.
/// Neighbours in the graph end up close in the order, so a pass over the graph
/// touches memory that is already in cache. It changes the speed and nothing
/// else. Returns ``order[new] = old``.
fn rcm_order(indptr: &[i64], indices: &[i32], n: usize) -> Vec<u32> {
    let degree: Vec<u32> = (0..n).map(|v| (indptr[v + 1] - indptr[v]) as u32).collect();
    let mut starts: Vec<u32> = (0..n as u32).collect();
    starts.par_sort_unstable_by_key(|&v| (degree[v as usize], v));
    let mut seen = vec![false; n];
    let mut order: Vec<u32> = Vec::with_capacity(n);
    let mut nbrs: Vec<u32> = Vec::new();
    for &s in &starts {
        if seen[s as usize] {
            continue;
        }
        seen[s as usize] = true;
        let mut head = order.len();
        order.push(s);
        while head < order.len() {
            let v = order[head] as usize;
            head += 1;
            nbrs.clear();
            for e in indptr[v] as usize..indptr[v + 1] as usize {
                let u = indices[e] as u32;
                if !seen[u as usize] {
                    seen[u as usize] = true;
                    nbrs.push(u);
                }
            }
            nbrs.sort_unstable_by_key(|&u| (degree[u as usize], u));
            order.extend_from_slice(&nbrs);
        }
    }
    order.reverse();
    order
}

/// The first level from the caller's arrays, renumbered by ``order``
/// (``order[new] = old``): integer weights, no diagonal, neighbours sorted.
/// Also the weight of each node's own loop. Rows are written in parallel.
fn build(indptr: &[i64], indices: &[i32], data: Weights, n: usize, wmax: f64, order: &[u32]) -> (Fine, Vec<u64>) {
    let unit = (1u64 << UNIT_BITS) as f64 / wmax;
    let mut pos = vec![0u32; n];
    for (new, &old) in order.iter().enumerate() {
        pos[old as usize] = new as u32;
    }
    let row_of = |new: usize| -> std::ops::Range<usize> {
        let old = order[new] as usize;
        indptr[old] as usize..indptr[old + 1] as usize
    };
    let len: Vec<usize> = (0..n).into_par_iter()
        .map(|new| {
            let old = order[new] as i32;
            row_of(new).filter(|&e| indices[e] != old).count()
        })
        .collect();
    let mut ip = Vec::with_capacity(n + 1);
    ip.push(0usize);
    for &l in &len {
        ip.push(ip.last().unwrap() + l);
    }
    let total = ip[n];
    let mut ix = vec![0u32; total];
    let mut wt = vec![0u32; total];
    let mut strength = vec![0u64; n];
    let mut self_w = vec![0u64; n];
    const ROWS: usize = 4096;
    // one block of rows, its slices of every output array
    let mut blocks = Vec::with_capacity(n.div_ceil(ROWS));
    {
        let (mut ix_r, mut wt_r): (&mut [u32], &mut [u32]) = (&mut ix, &mut wt);
        let (mut st_r, mut sw_r): (&mut [u64], &mut [u64]) = (&mut strength, &mut self_w);
        let mut lo = 0;
        while lo < n {
            let hi = (lo + ROWS).min(n);
            let m = ip[hi] - ip[lo];
            let (a, b) = ix_r.split_at_mut(m);
            let (c, d) = wt_r.split_at_mut(m);
            let (e, f) = st_r.split_at_mut(hi - lo);
            let (g, h) = sw_r.split_at_mut(hi - lo);
            blocks.push((lo, hi, a, c, e, g));
            ix_r = b;
            wt_r = d;
            st_r = f;
            sw_r = h;
            lo = hi;
        }
    }
    blocks.par_iter_mut().for_each_init(Vec::new, |row: &mut Vec<(u32, u32)>, (lo, hi, ix, wt, st, sw)| {
        let mut at = 0;
        for new in *lo..*hi {
            let old = order[new] as i32;
            row.clear();
            let (mut s, mut own) = (0u64, 0u64);
            for e in row_of(new) {
                let w = (data.get(e) * unit).round() as u32;
                s += w as u64;
                if indices[e] == old {
                    own += w as u64;
                } else {
                    row.push((pos[indices[e] as usize], w));
                }
            }
            row.sort_unstable_by_key(|x| x.0);
            for &(u, w) in row.iter() {
                ix[at] = u;
                wt[at] = w;
                at += 1;
            }
            st[new - *lo] = s;
            sw[new - *lo] = own;
        }
    });
    drop(blocks);
    (Fine { indptr: ip, indices: ix, weights: wt, strength, ident: order.to_vec() }, self_w)
}

/// The first pair of nodes whose two directions differ, if any: Leiden is
/// defined on an undirected graph, and on a directed one it would cluster
/// something, silently. A rounding of the weights is allowed for.
fn asymmetry(g: &Fine) -> Option<(u32, u32)> {
    let tol = ((1u64 << UNIT_BITS) as f64 * 1e-5) as i64;
    (0..g.n()).into_par_iter().find_map_first(|v| {
        for e in g.indptr[v]..g.indptr[v + 1] {
            let u = g.indices[e] as usize;
            let row = &g.indices[g.indptr[u]..g.indptr[u + 1]];
            let back = match row.binary_search(&(v as u32)) {
                Ok(i) => g.weights[g.indptr[u] + i] as i64,
                Err(_) => 0,
            };
            if (back - g.weights[e] as i64).abs() > tol {
                return Some((g.ident[v], g.ident[u]));
            }
        }
        None
    })
}

/// Leiden on the caller's arrays. Returns the labels in the caller's order
/// (the largest community 0), the modularity, and whether a local move ran
/// into the round guard.
#[allow(clippy::too_many_arguments)]
fn run(indptr: &[i64], indices: &[i32], data: Weights, n: usize, gamma: f64, n_iterations: i64,
       initial: Option<Vec<u32>>, max_levels: usize, reorder: bool, seed: Option<u64>, theta: f64,
       cancel: &AtomicBool)
       -> Result<(Vec<u32>, f64, bool), String> {
    let profile = std::env::var_os("PIASO_LEIDEN_PROFILE").is_some();
    let wmax = check(indptr, indices, data, n)?;
    if n == 0 {
        return Ok((Vec::new(), 0.0, false));
    }
    if !(wmax > 0.0) {
        return Ok(((0..n as u32).collect(), 0.0, false));
    }
    let t0 = std::time::Instant::now();
    let order: Vec<u32> = if reorder { rcm_order(indptr, indices, n) } else { (0..n as u32).collect() };
    let t_order = t0.elapsed().as_secs_f64();
    let (g, self_w) = build(indptr, indices, data, n, wmax, &order);
    let t_build = t0.elapsed().as_secs_f64() - t_order;
    if let Some((a, b)) = asymmetry(&g) {
        return Err(format!("the graph is not symmetric: the weights between nodes {} and {} differ in the two \
                            directions. Leiden clusters an undirected graph; pass the connectivities, or A + A.T", a, b));
    }
    if profile {
        eprintln!("leiden order [{:.2} GB]: {:.2}s, integer copy {:.2}s, symmetry check {:.2}s, {} threads", resident_gb(), t_order, t_build,
                  t0.elapsed().as_secs_f64() - t_order - t_build, rayon::current_num_threads());
    }
    let two_m = g.strength.iter().sum::<u64>() as f64;
    let params = Params {
        gamma,
        scale: gamma / two_m,
        seed: seed.unwrap_or(0),
        random: seed.is_some(),
        theta,
        unit: 1.0 / (1u64 << UNIT_BITS) as f64,
        profile,
        cancel,
    };
    // a negative count: repeat until a run leaves the partition unchanged
    let until_stable = n_iterations < 0;
    let runs = if until_stable { 1000 } else { n_iterations.max(1) as usize };
    let mut tally = Tally::default();
    let mut labels: Option<Vec<u32>> = initial.map(|l| order.iter().map(|&old| l[old as usize]).collect());
    if let Some(l) = labels.as_mut() {
        renumber(l);
    }
    for it in 0..runs {
        let next = leiden_once(&g, labels.as_deref(), &params, max_levels, it as u64, &mut tally);
        if cancel.load(Relaxed) {
            return Err(INTERRUPTED.into());
        }
        let stable = until_stable && labels.as_ref() == Some(&next);
        labels = Some(next);
        if stable {
            break;
        }
    }
    let labels = labels.unwrap();
    if n > 100_000 {
        give_back();
    }
    let q = modularity(&g, &labels, params.gamma, &self_w);
    let mut out = vec![0u32; n];
    for (new, &old) in order.iter().enumerate() {
        out[old as usize] = labels[new];
    }
    order_by_size(&mut out);
    if profile {
        eprintln!("leiden total: move {:.2}s refine {:.2}s aggregate {:.2}s, {} rounds, {:.2}s in all",
                  tally.t_move, tally.t_refine, tally.t_aggregate, tally.rounds, t0.elapsed().as_secs_f64());
    }
    Ok((out, q, tally.capped))
}

/// Leiden on a symmetric CSR matrix: parallel and deterministic.
///
/// Returns ``(membership, modularity)``: one community per node, the largest
/// numbered 0, and the modularity at ``resolution`` of that partition.
/// ``n_iterations`` runs the whole algorithm again from the previous result
/// (negative: until a run changes nothing, as igraph's ``-1``); its default is
/// ``piaso.tl.leiden``'s, 10, which always passes it (it was 2 here, so a
/// direct call and ``tl.leiden`` clustered differently);
/// ``initial`` starts from a given partition instead of singletons.
///
/// One graph and one ``seed`` give one partition, whatever ``n_threads``
/// (0: the threads of the process's pool) and whatever ``reorder``, which
/// renumbers the nodes along the graph (reverse Cuthill-McKee) so that each
/// pass touches memory in order. ``seed=None`` refines greedily, with no
/// random choice.
#[pyo3::pyfunction]
#[pyo3(signature = (indptr, indices, data, n_nodes, resolution=1.0, n_iterations=10, initial=None, max_levels=1000, reorder=true, seed=Some(0), theta=0.01, n_threads=0))]
#[allow(clippy::too_many_arguments)]
pub fn leiden_csr<'py>(
    py: pyo3::Python<'py>,
    indptr: numpy::PyReadonlyArray1<'py, i64>,
    indices: numpy::PyReadonlyArray1<'py, i32>,
    data: pyo3::Bound<'py, pyo3::PyAny>,
    n_nodes: usize,
    resolution: f64,
    n_iterations: i64,
    initial: Option<numpy::PyReadonlyArray1<'py, i64>>,
    max_levels: usize,
    reorder: bool,
    seed: Option<u64>,
    theta: f64,
    n_threads: usize,
) -> pyo3::PyResult<(pyo3::Bound<'py, numpy::PyArray1<i64>>, f64)> {
    use numpy::IntoPyArray;
    use pyo3::exceptions::PyValueError;
    use pyo3::types::PyAnyMethods;
    let (ip, ix) = (indptr.as_slice()?, indices.as_slice()?);
    // weights as they are, float32 or float64: no converted copy
    let d32: Option<numpy::PyReadonlyArray1<'py, f32>> = data.extract().ok();
    let d64: Option<numpy::PyReadonlyArray1<'py, f64>> = if d32.is_none() { Some(data.extract()?) } else { None };
    let s32: Option<&[f32]> = d32.as_ref().map(|a| a.as_slice()).transpose()?;
    let s64: Option<&[f64]> = d64.as_ref().map(|a| a.as_slice()).transpose()?;
    let init: Option<Vec<u32>> = match initial {
        Some(a) => {
            let a = a.as_slice()?;
            if a.len() != n_nodes || a.iter().any(|&x| x < 0) {
                return Err(PyValueError::new_err("initial must hold one non-negative label per node"));
            }
            let mut v: Vec<u32> = a.iter().map(|&x| x as u32).collect();
            renumber(&mut v);
            Some(v)
        }
        None => None,
    };
    if !(theta > 0.0) || !(resolution >= 0.0) {
        return Err(PyValueError::new_err("theta must be positive and resolution not negative"));
    }
    // Threads: what was asked for; else RAYON_NUM_THREADS; else the cores
    // this process may use (threads::count). A graph under SINGLE nodes is
    // clustered in a fraction of a second by one thread, and more only get in
    // each other's way (12,000 nodes: 0.27 s on one thread, 0.49 s on
    // twenty). The labels are the same whatever the number.
    let threads = if n_nodes < SINGLE { 1 } else { n_threads };
    let result = crate::threads::run_released(py, "leiden", threads, |cancel| {
        let w = match (s32, s64) {
            (Some(a), _) => Weights::F32(a),
            (None, Some(a)) => Weights::F64(a),
            _ => unreachable!(),
        };
        run(ip, ix, w, n_nodes, resolution, n_iterations, init, max_levels, reorder, seed, theta, cancel)
    })?;
    let (labels, q, capped) = result.map_err(PyValueError::new_err)?;
    if capped {
        pyo3::PyErr::warn(py, &py.get_type::<pyo3::exceptions::PyRuntimeWarning>(),
                          c"leiden: a local move was stopped after 1000 rounds; the partition is valid, and may be improved by more iterations", 1)?;
    }
    let out: Vec<i64> = labels.into_iter().map(|x| x as i64).collect();
    Ok((out.into_pyarray(py), q))
}

#[cfg(test)]
mod tests {
    use super::*;

    static NEVER: AtomicBool = AtomicBool::new(false);

    /// CSR arrays of an undirected graph given as (i, j, w), both directions
    /// added.
    fn csr(n: usize, edges: &[(u32, u32, f64)]) -> (Vec<i64>, Vec<i32>, Vec<f64>) {
        let mut rows: Vec<Vec<(i32, f64)>> = vec![Vec::new(); n];
        for &(i, j, w) in edges {
            rows[i as usize].push((j as i32, w));
            rows[j as usize].push((i as i32, w));
        }
        let mut indptr = vec![0i64];
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        for r in rows.iter_mut() {
            r.sort_by_key(|x| x.0);
            for &(j, w) in r.iter() {
                indices.push(j);
                data.push(w);
            }
            indptr.push(indices.len() as i64);
        }
        (indptr, indices, data)
    }

    fn two_cliques() -> (Vec<i64>, Vec<i32>, Vec<f64>) {
        // nodes 0-3 and 4-7 fully connected inside, one bridge 3-4
        let mut e = Vec::new();
        for block in [0u32, 4] {
            for i in block..block + 4 {
                for j in i + 1..block + 4 {
                    e.push((i, j, 1.0));
                }
            }
        }
        e.push((3, 4, 1.0));
        csr(8, &e)
    }

    /// `k` groups of `m` nodes: dense inside, sparse between, weights in
    /// (0, 1] with many exactly 1, as a neighbour graph has.
    fn planted(k: u32, m: u32, seed: u64) -> (usize, Vec<i64>, Vec<i32>, Vec<f64>) {
        let n = k * m;
        let mut e = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                let h = hash4(seed, 0, i as u64, j as u64);
                let same = i / m == j / m;
                let p = if same { 0.20 } else { 0.004 };
                if ((h >> 11) as f64 / (1u64 << 53) as f64) < p {
                    let w = if h & 3 == 0 { 1.0 } else { 0.05 + 0.9 * ((h & 0xFFFF) as f64 / 65536.0) };
                    e.push((i, j, w));
                }
            }
        }
        let (a, b, c) = csr(n as usize, &e);
        (n as usize, a, b, c)
    }

    fn go(g: &(Vec<i64>, Vec<i32>, Vec<f64>), n: usize, gamma: f64, reorder: bool, seed: Option<u64>, threads: usize) -> (Vec<u32>, f64) {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let (l, q, capped) = pool.install(|| run(&g.0, &g.1, Weights::F64(&g.2), n, gamma, 3, None, 1000, reorder, seed, 0.01, &NEVER)).unwrap();
        assert!(!capped);
        (l, q)
    }

    #[test]
    fn finds_two_cliques() {
        let g = two_cliques();
        for seed in [None, Some(0), Some(7)] {
            let (a, q) = go(&g, 8, 1.0, true, seed, 1);
            assert_eq!(a, vec![0, 0, 0, 0, 1, 1, 1, 1]);
            assert!((q - 0.4231).abs() < 1e-3, "{}", q);
        }
    }

    #[test]
    fn a_huge_resolution_leaves_singletons_and_a_tiny_one_merges_all() {
        let g = two_cliques();
        let (mut s, _) = go(&g, 8, 100.0, true, Some(0), 1);
        assert_eq!(renumber(&mut s), 8);
        let (mut t, _) = go(&g, 8, 1e-6, true, Some(0), 1);
        assert_eq!(renumber(&mut t), 1);
    }

    #[test]
    fn threads_and_layout_do_not_enter_the_result() {
        let (n, a, b, c) = planted(12, 500, 3);
        let g = (a, b, c);
        for seed in [Some(1), None] {
            let (base, q) = go(&g, n, 1.0, true, seed, 1);
            assert!(q > 0.5, "{}", q);
            for threads in [2, 4, 8] {
                assert_eq!(go(&g, n, 1.0, true, seed, threads).0, base, "{} threads", threads);
            }
            assert_eq!(go(&g, n, 1.0, false, seed, 4).0, base, "layout as given");
        }
    }

    #[test]
    fn the_planted_groups_are_found() {
        let (n, a, b, c) = planted(8, 400, 11);
        let (l, _) = go(&(a, b, c), n, 1.0, true, Some(0), 4);
        // every planted group falls into one community
        for grp in 0..8 {
            let mut count = std::collections::HashMap::new();
            for v in grp * 400..(grp + 1) * 400 {
                *count.entry(l[v]).or_insert(0usize) += 1;
            }
            assert!(*count.values().max().unwrap() >= 392, "group {}: {:?}", grp, count);
        }
    }

    #[test]
    fn edge_cases() {
        let none: (Vec<i64>, Vec<i32>, Vec<f64>) = (vec![0], vec![], vec![]);
        assert_eq!(go(&none, 0, 1.0, true, Some(0), 1).0, Vec::<u32>::new());
        // one node; isolated nodes; all weights zero
        assert_eq!(go(&(vec![0, 0], vec![], vec![]), 1, 1.0, true, Some(0), 1).0, vec![0]);
        let (l, _) = go(&csr(5, &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]), 5, 1.0, true, Some(0), 2);
        assert_eq!(l, vec![0, 0, 0, 1, 2]);
        let (mut z, q) = go(&csr(3, &[(0, 1, 0.0), (1, 2, 0.0)]), 3, 1.0, true, Some(0), 1);
        assert_eq!((renumber(&mut z), q), (3, 0.0));
        // a self-loop counts towards strength and is no edge; float32 weights
        let mut g = csr(4, &[(0, 1, 1.0), (2, 3, 1.0)]);
        g.1.insert(0, 0);
        g.2.insert(0, 5.0);
        for x in g.0.iter_mut().skip(1) {
            *x += 1;
        }
        assert_eq!(go(&g, 4, 1.0, true, Some(0), 1).0, vec![0, 0, 1, 1]);
        let f: Vec<f32> = g.2.iter().map(|&x| x as f32).collect();
        let (l32, _, _) = run(&g.0, &g.1, Weights::F32(&f), 4, 1.0, 2, None, 1000, true, Some(0), 0.01, &NEVER).unwrap();
        assert_eq!(l32, vec![0, 0, 1, 1]);
    }

    #[test]
    fn a_directed_graph_is_refused() {
        let mut g = csr(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
        g.2[0] = 0.5; // 0 -> 1 differs from 1 -> 0
        let e = run(&g.0, &g.1, Weights::F64(&g.2), 3, 1.0, 2, None, 1000, true, Some(0), 0.01, &NEVER).unwrap_err();
        assert!(e.contains("not symmetric"), "{}", e);
        // an edge present in one direction only
        let one_way = (vec![0i64, 1, 1], vec![1i32], vec![1.0f64]);
        assert!(run(&one_way.0, &one_way.1, Weights::F64(&one_way.2), 2, 1.0, 2, None, 1000, true, Some(0), 0.01, &NEVER).is_err());
    }

    #[test]
    fn an_initial_partition_is_the_starting_point() {
        let g = two_cliques();
        let init = Some(vec![0u32, 0, 0, 0, 1, 1, 1, 1]);
        let (l, _, _) = run(&g.0, &g.1, Weights::F64(&g.2), 8, 1.0, 1, init, 1000, true, Some(0), 0.01, &NEVER).unwrap();
        assert_eq!(l, vec![0, 0, 0, 0, 1, 1, 1, 1]);
    }
}
