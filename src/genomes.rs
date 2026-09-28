//! Chromosome sizes: the one table behind the tile grid, the peak caller and
//! `piaso.data.chrom_sizes`.
//!
//! A genome is either a name below or the path to a `chrom.sizes` (or `.fai`)
//! file. A file is reduced to its primary sequences by name: anything with an
//! underscore (`chr1_KI270706v1_random`, `chrUn_...`), anything starting
//! `chrUn`, and the mitochondrion are left out; everything else is kept, so a
//! reference that spells a chromosome `chr2A` keeps it. Applied to the
//! PIASO-data files, the rule returns exactly the tables below, so a genome
//! by name and the same genome by file agree by construction.
//!
//! The primate tables come from the Cell Ranger ARC references the fragments
//! are mapped against, so the names are the ones the fragments carry.

use std::fs;

pub type ChromSizes = Vec<(String, i64)>;

/// Genomes known by name, in the order `known_genomes` reports them.
pub const BUILTIN: &[(&str, &[(&str, i64)])] = &[
    ("hg38", &[
        ("chr1", 248956422), ("chr2", 242193529), ("chr3", 198295559),
        ("chr4", 190214555), ("chr5", 181538259), ("chr6", 170805979),
        ("chr7", 159345973), ("chr8", 145138636), ("chr9", 138394717),
        ("chr10", 133797422), ("chr11", 135086622), ("chr12", 133275309),
        ("chr13", 114364328), ("chr14", 107043718), ("chr15", 101991189),
        ("chr16", 90338345), ("chr17", 83257441), ("chr18", 80373285),
        ("chr19", 58617616), ("chr20", 64444167), ("chr21", 46709983),
        ("chr22", 50818468), ("chrX", 156040895), ("chrY", 57227415),
    ]),
    ("mm10", &[
        ("chr1", 195471971), ("chr2", 182113224), ("chr3", 160039680),
        ("chr4", 156508116), ("chr5", 151834684), ("chr6", 149736546),
        ("chr7", 145441459), ("chr8", 129401213), ("chr9", 124595110),
        ("chr10", 130694993), ("chr11", 122082543), ("chr12", 120129022),
        ("chr13", 120421639), ("chr14", 124902244), ("chr15", 104043685),
        ("chr16", 98207768), ("chr17", 94987271), ("chr18", 90702639),
        ("chr19", 61431566), ("chrX", 171031299), ("chrY", 91744698),
    ]),
    ("Mmul_10", &[
        ("chr1", 223616942), ("chr2", 196197964), ("chr3", 185288947),
        ("chr4", 169963040), ("chr5", 187317192), ("chr6", 179085566),
        ("chr7", 169868564), ("chr8", 145679320), ("chr9", 134124166),
        ("chr10", 99517758), ("chr11", 133066086), ("chr12", 130043856),
        ("chr13", 108737130), ("chr14", 128056306), ("chr15", 113283604),
        ("chr16", 79627064), ("chr17", 95433459), ("chr18", 74474043),
        ("chr19", 58315233), ("chr20", 77137495), ("chrX", 153388924),
        ("chrY", 11753682),
    ]),
    ("mCalJac1", &[
        ("chr1", 216975769), ("chr2", 202808461), ("chr3", 189455076),
        ("chr4", 173414057), ("chr5", 161716902), ("chr6", 159674559),
        ("chr7", 156129252), ("chr8", 126104592), ("chr9", 132900640),
        ("chr10", 136971485), ("chr11", 128529401), ("chr12", 123697827),
        ("chr13", 117638787), ("chr14", 112969529), ("chr15", 98621277),
        ("chr16", 98110366), ("chr17", 74700100), ("chr18", 47063576),
        ("chr19", 50540917), ("chr20", 44412365), ("chr21", 50614742),
        ("chr22", 49900457), ("chrX", 146897247), ("chrY", 4974773),
    ]),
    ("hg19", &[
        ("chr1", 249250621), ("chr2", 243199373), ("chr3", 198022430),
        ("chr4", 191154276), ("chr5", 180915260), ("chr6", 171115067),
        ("chr7", 159138663), ("chr8", 146364022), ("chr9", 141213431),
        ("chr10", 135534747), ("chr11", 135006516), ("chr12", 133851895),
        ("chr13", 115169878), ("chr14", 107349540), ("chr15", 102531392),
        ("chr16", 90354753), ("chr17", 81195210), ("chr18", 78077248),
        ("chr19", 59128983), ("chr20", 63025520), ("chr21", 48129895),
        ("chr22", 51304566), ("chrX", 155270560), ("chrY", 59373566),
    ]),
    ("mm39", &[
        ("chr1", 195154279), ("chr2", 181755017), ("chr3", 159745316),
        ("chr4", 156860686), ("chr5", 151758149), ("chr6", 149588044),
        ("chr7", 144995196), ("chr8", 130127694), ("chr9", 124359700),
        ("chr10", 130530862), ("chr11", 121973369), ("chr12", 120092757),
        ("chr13", 120883175), ("chr14", 125139656), ("chr15", 104073951),
        ("chr16", 98008968), ("chr17", 95294699), ("chr18", 90720763),
        ("chr19", 61420004), ("chrX", 169476592), ("chrY", 91455967),
    ]),
];

/// The names a caller may pass instead of a file.
pub fn known_genomes() -> Vec<&'static str> {
    BUILTIN.iter().map(|(g, _)| *g).collect()
}

/// The table for a genome known by name.
pub fn builtin(genome: &str) -> Option<ChromSizes> {
    BUILTIN.iter().find(|(g, _)| *g == genome)
        .map(|(_, t)| t.iter().map(|(c, n)| (c.to_string(), *n)).collect())
}

/// Whether a sequence name from a chrom.sizes file is a primary chromosome.
pub fn is_primary_chrom(name: &str) -> bool {
    !(name.contains('_') || name.starts_with("chrUn")
      || matches!(name, "chrM" | "chrMT" | "MT" | "M"))
}

/// Canonical order: numbered chromosomes by number (a letter suffix such as
/// `2A` after its number), then X, Y, then anything else by name.
pub fn chrom_sort_key(name: &str) -> (u8, u32, String) {
    let s = name.strip_prefix("chr").unwrap_or(name);
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    if !digits.is_empty() {
        return (0, digits.parse().unwrap_or(u32::MAX), s[digits.len()..].to_string());
    }
    match s {
        "X" => (1, 0, String::new()),
        "Y" => (1, 1, String::new()),
        _ => (2, 0, s.to_string()),
    }
}

/// Read a chrom.sizes / .fai file down to its primary chromosomes, in canonical order.
pub fn from_file(path: &str) -> Result<ChromSizes, String> {
    let content = fs::read_to_string(path).map_err(|e| format!(
        "'{}' is not a genome known by name ({}) and could not be read as a chrom.sizes file: {}",
        path, known_genomes().join(", "), e))?;
    let mut sizes: ChromSizes = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let mut parts = line.split(['\t', ' ']).filter(|p| !p.is_empty());
        let (Some(chrom), Some(size)) = (parts.next(), parts.next()) else { continue };
        if !is_primary_chrom(chrom) { continue; }
        let size: i64 = size.parse().map_err(|_| format!("Invalid size '{}' for chrom '{}' in '{}'", size, chrom, path))?;
        sizes.push((chrom.to_string(), size));
    }
    if sizes.is_empty() {
        return Err(format!("No primary chromosomes found in '{}' (names with an underscore, chrUn* and the \
                            mitochondrion are left out; the rest must have a tab-separated size).", path));
    }
    sizes.sort_by_key(|(c, _)| chrom_sort_key(c));
    Ok(sizes)
}

/// A genome by name or by file.
pub fn chrom_sizes(genome: &str) -> Result<ChromSizes, String> {
    match builtin(genome) {
        Some(t) => Ok(t),
        None => from_file(genome),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tables_hold_the_primary_chromosomes() {
        let n: Vec<(&str, usize)> = BUILTIN.iter().map(|(g, t)| (*g, t.len())).collect();
        assert_eq!(n, vec![("hg38", 24), ("mm10", 21), ("Mmul_10", 22), ("mCalJac1", 24), ("hg19", 24), ("mm39", 21)]);
        for (_, t) in BUILTIN {
            assert!(t.iter().all(|(c, n)| is_primary_chrom(c) && *n > 0));
            let mut names: Vec<&str> = t.iter().map(|(c, _)| *c).collect();
            let sorted = { let mut s = names.clone(); s.sort_by_key(|c| chrom_sort_key(c)); s };
            assert_eq!(names, sorted, "tables are stored in canonical order");
            names.dedup(); assert_eq!(names.len(), t.len());
        }
    }

    #[test]
    fn a_file_keeps_lettered_chromosomes_and_drops_the_rest() {
        let dir = std::env::temp_dir().join(format!("piaso_genomes_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("panTro.chrom.sizes");
        std::fs::write(&p, "chr2B\t100\nchrUn_NW_1\t5\nchr1\t300\nchr2A\t200\nchrM\t16\nchrX\t150\nchr1_random\t7\nchrY\t50\n#c\t1\n").unwrap();
        let t = from_file(p.to_str().unwrap()).unwrap();
        let names: Vec<&str> = t.iter().map(|(c, _)| c.as_str()).collect();
        assert_eq!(names, vec!["chr1", "chr2A", "chr2B", "chrX", "chrY"]);
        assert!(matches!(chrom_sizes("no_such_genome_or_file"), Err(e) if e.contains("hg38, mm10")));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_piaso_data_files_reduce_to_the_tables() {
        // Only where the checkout has PIASO-data next to it (the developer tree); else nothing to check.
        for (g, _) in BUILTIN {
            let p = format!("{}/../PIASO-data/{}/{}.chrom.sizes", env!("CARGO_MANIFEST_DIR"), g, g);
            if !std::path::Path::new(&p).exists() { continue; }
            assert_eq!(from_file(&p).unwrap(), builtin(g).unwrap(), "{}", g);
        }
    }
}
