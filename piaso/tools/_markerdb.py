"""
PIASOmarkerDB Client - Cell Type Marker Gene Database
======================================================

A comprehensive Python client for accessing PIASOmarkerDB, a database of cell type 
marker genes with standardized specificity scores across various tissues, species, 
studies, and conditions.

PIASOmarkerDB URL: https://piaso.org/piasomarkerdb/

Usage
-----
    import piaso
    
    # Query marker genes
    df = piaso.tl.queryPIASOmarkerDB(gene="Foxp2", species="Mouse")
    
    # Get the marker dict, in the form kamon takes
    marker_dict = piaso.tl.queryPIASOmarkerDB(
        study="AllenWholeMouseBrain2023_isocortex",
        species="Mouse",
        as_dict=True
    )
    
    # List available studies
    studies = piaso.tl.queryPIASOmarkerDB(list_studies=True)

    # Marker regions, named as a PIASO store's tiles: the dict that names nuclei from their tile counts
    regions = piaso.tl.getMarkerRegions(
        study="ZuRen2023_WholeBrain_ATAC",
        genome="mm10",
        top_n=200,
        as_dict=True
    )
    
    # Analyze gene lists for cell type inference
    df = piaso.tl.analyzeMarkers(["Syt6", "Tle4", "Foxp2", "Fezf2"])
    
    # Analyze COSG results (DataFrame input) with specific study
    import pandas as pd
    cosg_df = pd.DataFrame(adata.uns['cosg']['names']).head(50)
    results, top_hits = piaso.tl.analyzeMarkers(
        cosg_df, 
        n_top_genes=50,
        species="Mouse",
        studies="AllenWholeMouseBrain2023_isocortex"
    )
"""

from __future__ import annotations

import hashlib
import json
import numbers
import os
import time
import warnings
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter

try:                      # requests re-exports urllib3's Retry; urllib3 is not a declared dependency
    from requests.adapters import Retry
except ImportError:       # pragma: no cover - a requests without it retries connections only
    Retry = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# =============================================================================
# Exceptions (Internal Use Only - Not Exported)
# =============================================================================


# ``as_dict=True`` used to return ``(DataFrame, dict)``. Almost every use of it
# wanted the dict -- it is what feeds kamon -- and paid for it
# by unpacking a two-element tuple whose first element was then thrown away. It
# now returns the dict alone. ``as_dict='both'`` is the old behaviour, kept for
# one release so existing code has somewhere to go.
_AS_DICT_BOTH = "both"
_as_dict_change_announced = False


def _resolve_as_dict(as_dict):
    """Return ``(want_dict, want_frame)`` for an ``as_dict`` argument."""
    global _as_dict_change_announced
    if as_dict is False or as_dict is None:
        return False, True
    if as_dict is True:
        if not _as_dict_change_announced:
            _as_dict_change_announced = True
            warnings.warn(
                "as_dict=True now returns the marker dictionary alone; it used "
                "to return (DataFrame, dict). Unpacking the result into two "
                "names will fail, or -- with exactly two cell types -- quietly "
                "give you two cell-type names. Use as_dict='both' for the old "
                "pair (deprecated, removed next release), or drop the unpacking.",
                FutureWarning, stacklevel=3,
            )
        return True, False
    if isinstance(as_dict, str) and as_dict.lower() == _AS_DICT_BOTH:
        warnings.warn(
            "as_dict='both' is deprecated and will be removed in the next "
            "release. Call twice, or take the dict with as_dict=True and build "
            "the frame from your own query.",
            DeprecationWarning, stacklevel=3,
        )
        return True, True
    raise ValueError(
        f"as_dict={as_dict!r} is not valid: pass True for the marker "
        f"dictionary, False for the DataFrame, or 'both' for the deprecated "
        f"(DataFrame, dict) pair.")


class PIASOmarkerDBError(Exception):
    """Base exception for PIASOmarkerDB errors."""
    pass


class APIError(PIASOmarkerDBError):
    """API request error."""
    def __init__(
        self, 
        message: str, 
        status_code: int = None, 
        response: dict = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class ValidationError(PIASOmarkerDBError):
    """Input validation error."""
    pass


class MarkerDBConnectionError(PIASOmarkerDBError):
    """Connection error to PIASOmarkerDB server."""
    pass


class AuthenticationError(PIASOmarkerDBError):
    """Authentication error (kept for potential future use)."""
    pass


# =============================================================================
# Marker regions (Internal Use Only - Not Exported)
# =============================================================================

#: A marker region's fields, in the server's order (PIASOmarkerDB, regions.py: COLUMNS). The conservation fields
#: come last and are empty where the database holds none.
_REGION_COLUMNS = [
    "cell_type", "region", "chrom", "start", "end", "cosg_score", "cosg_rank", "study_publication", "species",
    "tissue", "condition", "genome", "ccre_class", "ccre_accession", "nearest_gene", "tss_distance", "nearby_genes",
    "phylop_mean", "phastcons_mean", "bases_scored", "ortholog", "ortholog_markers", "ortholog_contexts",
]
_REGION_STUDY_COLUMNS = ["study_publication", "species", "genome", "tissue", "condition", "cell_types", "regions"]
#: The genome build of each species' marker regions: the database keeps one build per species.
_REGION_GENOMES = {"Human": "hg38", "Mouse": "mm10"}
#: The most rows the server returns in one JSON page.
_REGION_PAGE_MAX = 100_000
#: The most marker rows the server returns for one request with a limit (/api/v1/markers caps ``limit`` there,
#: so a larger page came back short and was read as the last one).
_MARKER_PAGE_MAX = 100_000
#: Genes per request when gene lists are looked up together (analyzeGenes): keeps each URL short.
_GENE_BATCH = 200


def _region_genome(value, what):
    """``'Human'``, ``'human'``, ``'hg38'`` -> ``'hg38'``, the same for Mouse and mm10; ``None`` stays ``None``."""
    if value is None:
        return None
    v = str(value).strip().lower()
    for species, build in _REGION_GENOMES.items():
        if v in (species.lower(), build):
            return build
    raise ValidationError(
        f"{what}={value!r}: PIASOmarkerDB holds marker regions for Human (hg38) and Mouse (mm10) only.")


def _tile_ids(chrom, start, end, tile_size):
    """The tiles of a PIASO store's grid that ``[start, end)`` overlaps, named as the importer names them:
    ``'chr1:3000500-3001000'``, the tile's start and its start plus ``tile_size`` (0-based, half-open). A
    marker region is one 500 bp tile of that grid, so at the default tile size it maps to exactly one tile."""
    lo, hi = start // tile_size, -(-end // tile_size)
    return [f"{chrom}:{k * tile_size}-{(k + 1) * tile_size}" for k in range(lo, max(hi, lo + 1))]


def _positive_int(name, value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValidationError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


# =============================================================================
# Main Client Class
# =============================================================================

class PIASOmarkerDB:
    """
    Python client for accessing PIASOmarkerDB.
    
    PIASOmarkerDB is a comprehensive database of cell type marker genes with 
    specificity scores across various tissues, species, studies, and conditions,
    powered by PIASO (Precise Integrative Analysis of Single-cell Omics) methodologies.
    
    Parameters
    ----------
    base_url : str, optional
        Base URL for the PIASOmarkerDB API. 
        Default: "https://piaso.org/piasomarkerdb"
    timeout : int, optional
        Request timeout in seconds. Default: 30
    cache_dir : str or Path, optional
        A directory to keep the server's answers in, one JSON file per request, reused for ``cache_ttl``
        seconds (repeated analyses then send no request). Default ``None``: nothing is cached and no directory
        is created. (It defaulted to ~/.piaso/markers, created on every client although nothing was ever
        cached there, so a read-only home directory broke every call.)
    cache_ttl : float, optional
        Seconds a cached answer is reused. Default: 86400 (a day); ``None`` keeps them until deleted.
    retries : int, optional
        Retries, with exponential backoff, of a request that fails to connect or is answered 429, 500, 502,
        503 or 504 (honouring Retry-After). Default: 3.
    
    Examples
    --------
    Basic usage:
    
    >>> from piaso.tools import PIASOmarkerDB
    >>> client = PIASOmarkerDB()
    >>> 
    >>> # Query markers
    >>> df = client.getMarkers(gene="Foxp2")
    >>> 
    >>> # Get as dict
    >>> marker_dict = client.getMarkers(
    ...     study="AllenWholeMouseBrain2023_isocortex",
    ...     as_dict=True
    ... )
    >>> 
    >>> # Download markers
    >>> client.downloadMarkers("markers.csv", species="Mouse")
    
    See Also
    --------
    piaso.tl.queryPIASOmarkerDB : Main query function
    piaso.tl.analyzeMarkers : Gene list analysis
    
    Notes
    -----
    PIASOmarkerDB provides marker genes with specificity scores computed using
    PIASO's standardized methodology across multiple single-cell RNA-seq studies.
    
    Website: https://piaso.org/piasomarkerdb/
    """
    
    # Class constants
    DEFAULT_BASE_URL = "https://piaso.org/piasomarkerdb"
    API_VERSION = "v1"
    DEFAULT_TIMEOUT = 30
    DEFAULT_RETRIES = 3
    
    # Return value for no matches
    NO_MATCH_VALUE = "Unassigned"
    
    def __init__(
        self,
        base_url: str = None,
        timeout: int = None,
        cache_dir: str | Path | None = None,
        cache_ttl: Optional[float] = 86400,
        retries: int = None,
    ):
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip('/')
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self._session = requests.Session()
        # transient failures are retried with backoff (none were: one dropped connection or a 503 from a
        # restarting server failed the whole analysis)
        n = self.DEFAULT_RETRIES if retries is None else int(retries)
        adapter = HTTPAdapter(max_retries=n if Retry is None else Retry(
            total=n, connect=n, read=n, status=n, backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset({"GET", "HEAD"}),
            raise_on_status=False, respect_retry_after_header=True))
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._aliases = None            # {former study name: current name}, read once (studyAliases)
        # an opt-in response cache; the directory is made on the first write
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl = cache_ttl
    
    def __repr__(self) -> str:
        return f"PIASOmarkerDB(base_url='{self.base_url}')"
    
    def _build_url(self, endpoint: str) -> str:
        """Build full API URL."""
        return f"{self.base_url}/api/{self.API_VERSION}/{endpoint.lstrip('/')}"
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict = None,
        **kwargs
    ) -> Union[dict, list]:
        """
        Make API request.
        
        Parameters
        ----------
        method : str
            HTTP method (GET, POST, etc.)
        endpoint : str
            API endpoint
        params : dict, optional
            Query parameters
        **kwargs
            Additional arguments passed to requests
            
        Returns
        -------
        dict or list
            JSON response data (can be dict or list depending on endpoint)
        """
        url = self._build_url(endpoint)
        key = None
        if self.cache_dir is not None and method.upper() == "GET" and not kwargs:
            key = self._cache_key(url, params)
            cached = self._cache_read(key)
            if cached is not None:
                return cached["data"]

        try:
            response = self._session.request(
                method=method,
                url=url,
                params=params,
                timeout=self.timeout,
                **kwargs
            )
            
            if response.status_code >= 400:
                try:
                    error_data = response.json()
                    if isinstance(error_data, dict):
                        message = error_data.get('error', response.text)
                    else:
                        message = response.text
                except Exception:
                    message = response.text
                raise APIError(message, status_code=response.status_code)

            try:
                data = response.json()
            except ValueError as e:
                # a proxy's or a maintenance page's HTML: say so, not a raw JSONDecodeError
                raise APIError(
                    f"{url} did not answer with JSON (status {response.status_code}, content type "
                    f"{response.headers.get('Content-Type')!r}): {response.text[:200]!r}",
                    status_code=response.status_code) from e
            if key is not None:
                self._cache_write(key, url, params, data)
            return data
            
        except requests.exceptions.Timeout:
            raise MarkerDBConnectionError(
                f"Request timed out after {self.timeout}s"
            )
        except requests.exceptions.ConnectionError as e:
            raise MarkerDBConnectionError(
                f"Failed to connect to {url}: {e}"
            )
    
    def _cache_key(self, url: str, params: Optional[dict]) -> str:
        blob = json.dumps([url, sorted((str(k), str(v)) for k, v in (params or {}).items())])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_read(self, key: str) -> Optional[dict]:
        path = self.cache_dir / f"{key}.json"
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            return None
        if not isinstance(rec, dict) or "data" not in rec:
            return None
        if self.cache_ttl is not None and time.time() - float(rec.get("time", 0)) > self.cache_ttl:
            return None
        return rec

    def _cache_write(self, key: str, url: str, params: Optional[dict], data) -> None:
        path = self.cache_dir / f"{key}.json"
        tmp = path.with_name(f".{key}.{os.getpid()}.tmp")
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"url": url, "params": {str(k): str(v) for k, v in (params or {}).items()},
                           "time": time.time(), "data": data}, fh)
            os.replace(tmp, path)
        except OSError as exc:
            warnings.warn(f"PIASOmarkerDB: the response could not be cached in {self.cache_dir} ({exc})",
                          UserWarning, stacklevel=3)

    def _extract_data(self, response: Union[dict, list]) -> list:
        """
        Extract data from API response.
        
        Handles both formats:
        - Direct list: [item1, item2, ...]
        - Dict with data key: {'data': [item1, item2, ...]}
        
        Parameters
        ----------
        response : dict or list
            API response
            
        Returns
        -------
        list
            Extracted data list
        """
        if isinstance(response, list):
            return response
        elif isinstance(response, dict):
            return response.get('data', [])
        else:
            return []
    
    # =========================================================================
    # Core Query Methods
    # =========================================================================
    
    def getMarkers(
        self,
        gene: Union[str, List[str]] = None,
        cell_type: Union[str, List[str]] = None,
        study: str = None,
        species: str = None,
        tissue: str = None,
        condition: str = None,
        min_score: float = None,
        max_score: float = None,
        limit: int = None,
        offset: int = 0,
        as_dict: Union[bool, str] = False,
    ) -> Union['pd.DataFrame', Tuple['pd.DataFrame', Dict[str, List[str]]]]:
        """
        Query markers with flexible filters.
        
        Parameters
        ----------
        gene : str or list of str, optional
            Gene symbol(s) to filter by.
        cell_type : str or list of str, optional
            Cell type(s) to filter by.
        study : str, optional
            Study/publication identifier to filter by.
        species : str, optional
            Species to filter by (e.g., "Human", "Mouse").
        tissue : str, optional
            Tissue to filter by (e.g., "Blood", "Brain").
        condition : str, optional
            Experimental condition to filter by.
        min_score : float, optional
            Minimum specificity score (must be non-negative).
        max_score : float, optional
            Maximum specificity score (must be non-negative).
        limit : int, optional
            Maximum number of results. Default: None (no limit).
        offset : int, optional
            Number of results to skip (for pagination). Default: 0.
        as_dict : bool or {'both'}, optional
            ``True`` returns the marker dictionary ``{cell_type: [genes]}``
            alone — the form that feeds ``kamon``. ``False``
            (the default) returns the DataFrame. ``'both'`` returns the old
            ``(DataFrame, dict)`` pair and is deprecated.

        Returns
        -------
        pd.DataFrame
            Query results (as_dict=False).

        dict
            ``{cell_type: [genes]}`` (as_dict=True).

        tuple (pd.DataFrame, dict)
            Deprecated; as_dict='both'.

        Examples
        --------
        >>> df = client.getMarkers(gene="Foxp2", species="Mouse")
        >>> marker_dict = client.getMarkers(
        ...     study="AllenWholeMouseBrain2023_isocortex",
        ...     as_dict=True
        ... )
        """
        params, local_ct = self._marker_params(gene, cell_type, study, species, tissue, condition,
                                               min_score, max_score)
        rows = self._marker_rows(params, local_ct, limit=limit, offset=offset)

        # Convert to DataFrame
        if not HAS_PANDAS:
            raise ImportError(
                "pandas is required for PIASOmarkerDB. "
                "Install with: pip install pandas"
            )

        df = pd.DataFrame(rows)

        want_dict, want_frame = _resolve_as_dict(as_dict)
        if want_dict:
            marker_dict = self._to_marker_dict(df)
            return (df, marker_dict) if want_frame else marker_dict
        return df

    def _marker_params(self, gene=None, cell_type=None, study=None, species=None, tissue=None,
                       condition=None, min_score=None, max_score=None):
        """The /markers query parameters, and the cell types to keep here (``None``: the server filters).

        The server splits ``cell_type`` on commas, so a cell type whose name holds one could not be asked
        for; a query naming one is sent without ``cell_type`` and filtered here, by exact name."""
        params = {}
        if gene:
            params['gene'] = ','.join(gene) if isinstance(gene, list) else gene
        local_ct = None
        if cell_type:
            cts = [cell_type] if isinstance(cell_type, str) else [str(c) for c in cell_type]
            if any(',' in c for c in cts):
                local_ct = set(cts)
            else:
                params['cell_type'] = ','.join(cts)
        if study:
            params['study'] = study
        if species:
            params['species'] = species
        if tissue:
            params['tissue'] = tissue
        if condition:
            params['condition'] = condition
        if min_score is not None:
            if min_score < 0:
                raise ValidationError("min_score must be non-negative")
            params['min_score'] = min_score
        if max_score is not None:
            if max_score < 0:
                raise ValidationError("max_score must be non-negative")
            params['max_score'] = max_score
        return params, local_ct

    def _marker_rows(self, params, local_ct=None, limit=None, offset=0):
        """Rows of /markers for ``params``: every row (``limit=None``: one request without a limit, which the
        server answers whole), or ``limit`` rows from ``offset``, asked for in pages within the server's cap
        (``_MARKER_PAGE_MAX``; a larger limit came back short, silently). Rows of ``local_ct`` only, when
        given."""
        if local_ct is not None:
            rows = [r for r in self._extract_data(self._request('GET', 'markers', params=dict(params)))
                    if str(r.get('cell_type')) in local_ct]
            rows = rows[offset:]
            return rows if limit is None else rows[:limit]
        if limit is None:
            # the server writes OFFSET without LIMIT then, which SQLite refuses: slice here
            rows = self._extract_data(self._request('GET', 'markers', params=dict(params)))
            return rows[offset:] if offset > 0 else rows
        rows, want = [], int(limit)
        while len(rows) < want:
            n = min(want - len(rows), _MARKER_PAGE_MAX)
            q = dict(params, limit=n)
            if offset + len(rows) > 0:
                q['offset'] = offset + len(rows)
            response = self._request('GET', 'markers', params=q)
            page = self._extract_data(response)
            rows += page
            has_more = response.get('has_more') if isinstance(response, dict) else None
            if not page or has_more is False or (has_more is None and len(page) < n):
                break
        return rows

    def getAllMarkers(
        self,
        batch_size: int = 10000,
        verbose: bool = False,
        as_dict: Union[bool, str] = False,
        **kwargs
    ) -> Union['pd.DataFrame', Tuple['pd.DataFrame', Dict[str, List[str]]]]:
        """
        Get all markers matching filters with automatic pagination.
        
        Parameters
        ----------
        batch_size : int, optional
            Number of records per request, at most 100,000 (the server's cap). Default: 10000.
        verbose : bool, optional
            Print progress information. Default: False.
        as_dict : bool or {'both'}, optional
            ``True`` returns the marker dictionary alone, ``False`` the
            DataFrame, ``'both'`` the deprecated pair. See ``getMarkers``.
        **kwargs
            Same filter parameters as getMarkers().

        Returns
        -------
        pd.DataFrame, dict, or tuple
            All matching markers, in the form ``as_dict`` asks for.
        """
        kwargs = dict(kwargs)
        kwargs.pop('as_dict', None)
        kwargs.pop('limit', None)
        kwargs.pop('offset', None)
        params, local_ct = self._marker_params(**kwargs)
        # the server answers at most _MARKER_PAGE_MAX rows a page: a larger batch came back short and was
        # read as the last page
        batch_size = min(int(batch_size), _MARKER_PAGE_MAX)

        all_data = []
        if local_ct is not None:
            all_data = self._marker_rows(params, local_ct)
        else:
            offset = 0
            batch_num = 0
            while True:
                batch_num += 1
                q = dict(params, limit=batch_size)
                if offset > 0:
                    q['offset'] = offset
                response = self._request('GET', 'markers', params=q)
                page = self._extract_data(response)
                if not page:
                    break
                all_data += page
                if verbose:
                    print(f"Fetching markers... batch {batch_num} ({len(page)} records)")
                has_more = response.get('has_more') if isinstance(response, dict) else None
                if has_more is False or (has_more is None and len(page) < batch_size):
                    break
                offset += len(page)

        df = pd.DataFrame(all_data)
        if verbose:
            print(f"Total: {len(df)} markers")

        want_dict, want_frame = _resolve_as_dict(as_dict)
        if want_dict:
            marker_dict = self._to_marker_dict(df)
            return (df, marker_dict) if want_frame else marker_dict
        return df

    def listStudies(
        self,
        species: str = None,
        tissue: str = None,
    ) -> List[str]:
        """
        Get list of available study keys.
        
        Parameters
        ----------
        species : str, optional
            Filter by species.
        tissue : str, optional
            Filter by tissue.
        
        Returns
        -------
        list of str
            List of study/publication identifiers.
        
        Examples
        --------
        >>> studies = client.listStudies()
        >>> print(f"Total studies: {len(studies)}")
        >>> mouse_studies = client.listStudies(species="Mouse")
        """
        params = {}
        if species:
            params['species'] = species
        if tissue:
            params['tissue'] = tissue
        
        response = self._request('GET', 'studies', params=params)
        data = self._extract_data(response)
        
        # Handle different response formats
        studies = []
        for item in data:
            if isinstance(item, str):
                # Direct string
                studies.append(item)
            elif isinstance(item, dict):
                # Dict with study info
                study_name = item.get('study_publication') or item.get('study') or item.get('name', '')
                if study_name:
                    studies.append(study_name)
        
        return studies

    def studyAliases(self) -> Dict[str, str]:
        """
        Former study names and the names the studies carry now.

        PIASOmarkerDB renames a study whose name breaks its naming rules and keeps the former name answering on the
        server. This returns ``{former name: current name}``, read once per client (empty from a server that keeps
        no aliases). ``analyzeGenes`` and ``analyzeMarkers`` resolve study names through it before checking them,
        so code written with a former name keeps working.

        Returns
        -------
        dict
            Former name to current name.

        Examples
        --------
        >>> client.studyAliases()['SEAAD2024_MTG_Subclass']
        'SEAAD2024_MTG_subclass'
        """
        if self._aliases is None:
            try:
                data = self._extract_data(self._request('GET', 'study-aliases'))
            except APIError as e:
                if e.status_code != 404:            # a server from before the aliases has no such endpoint
                    raise
                data = []
            self._aliases = {str(d['old_name']): str(d['new_name']) for d in data
                             if isinstance(d, dict) and d.get('old_name') and d.get('new_name')}
        return dict(self._aliases)

    def _resolve_studies(self, names: Optional[List[str]]) -> Optional[List[str]]:
        """Each study name as the study is called now: a former name (in any case) becomes its current name, with a
        warning; other names are returned as given."""
        if not names:
            return names
        aliases = self.studyAliases()
        if not aliases:
            return list(names)
        folded = {k.lower(): v for k, v in aliases.items()}
        out = []
        for s in names:
            new = aliases.get(s) or folded.get(str(s).lower())
            if new and new != s:
                warnings.warn(f"Study '{s}' was renamed to '{new}' in PIASOmarkerDB; using '{new}'.",
                              UserWarning, stacklevel=3)
                s = new
            out.append(s)
        return out
    
    def getCellTypes(
        self,
        species: str = None,
        tissue: str = None,
        study: str = None,
    ) -> List[str]:
        """
        Get list of available cell types.
        
        Parameters
        ----------
        species : str, optional
            Filter by species.
        tissue : str, optional
            Filter by tissue.
        study : str, optional
            Filter by study.
        
        Returns
        -------
        list of str
            List of cell type names.
        """
        params = {}
        if species:
            params['species'] = species
        if tissue:
            params['tissue'] = tissue
        if study:
            params['study'] = study
        
        response = self._request('GET', 'cell-types', params=params)
        data = self._extract_data(response)
        
        # Handle different response formats
        cell_types = []
        for item in data:
            if isinstance(item, str):
                cell_types.append(item)
            elif isinstance(item, dict):
                ct_name = item.get('cell_type') or item.get('name', '')
                if ct_name:
                    cell_types.append(ct_name)
        
        # Return unique sorted list
        return sorted(list(set(cell_types)))
    
    def getGenes(
        self,
        cell_type: str = None,
        species: str = None,
        tissue: str = None,
        min_score: float = None,
    ) -> List[str]:
        """
        Get list of unique gene symbols.
        
        Parameters
        ----------
        cell_type : str, optional
            Filter by cell type.
        species : str, optional
            Filter by species.
        tissue : str, optional
            Filter by tissue.
        min_score : float, optional
            Minimum specificity score.
        
        Returns
        -------
        list of str
            List of unique gene symbols.
        """
        params = {}
        if cell_type:
            params['cell_type'] = cell_type
        if species:
            params['species'] = species
        if tissue:
            params['tissue'] = tissue
        if min_score is not None:
            if min_score < 0:
                raise ValidationError("min_score must be non-negative")
            params['min_score'] = min_score
        
        response = self._request('GET', 'genes', params=params)
        data = self._extract_data(response)
        
        # Handle different response formats
        genes = []
        for item in data:
            if isinstance(item, str):
                genes.append(item)
            elif isinstance(item, dict):
                gene_name = item.get('gene') or item.get('name', '')
                if gene_name:
                    genes.append(gene_name)
        
        return genes
    
    # =========================================================================
    # Gene List Analysis
    # =========================================================================
    
    def analyzeGenes(
        self,
        genes: Union[List[str], 'pd.DataFrame', Dict[str, List[str]]],
        n_top_genes: int = 50,
        species: str = None,
        tissue: str = None,
        studies: Union[str, List[str]] = None,
        min_genes: int = 1,
        exclude_cell_types: List[str] = None,
        exclude_studies: List[str] = None,
    ) -> Union['pd.DataFrame', Tuple[Dict[str, 'pd.DataFrame'], Dict[str, str]]]:
        """
        Analyze gene list(s) to infer potential cell types.
        
        Parameters
        ----------
        genes : list of str, pd.DataFrame, or dict
            Gene input. Supports three formats:
            
            - **list of str**: Single list of gene symbols. Returns DataFrame.
            - **pd.DataFrame**: Columns are clusters/cell types, rows are genes.
              E.g., COSG output: pd.DataFrame(adata.uns['cosg']['names']).head(50).
              Returns tuple (results_dict, top_hits_dict).
            - **dict**: {cluster_name: [gene_list]}. 
              Returns tuple (results_dict, top_hits_dict).
              
        n_top_genes : int, optional
            For DataFrame input: only use top N genes per column. Default: 50.
            Useful for COSG results which may have many genes ranked.
        species : str, optional
            Filter results by species (e.g., "Human", "Mouse").
        tissue : str, optional
            Filter results by tissue.
        studies : str or list of str, optional
            Study/studies to include in analysis. Only cell types from these
            studies will be considered. Study names are validated against
            PIASOmarkerDB; a renamed study's former name is resolved to its
            current name, with a warning (``studyAliases``). Cannot overlap with exclude_studies.
            Default: None (use all studies).
        min_genes : int, optional
            Minimum number of genes that must match a cell type. Default: 1.
        exclude_cell_types : list of str, optional
            Cell types to exclude from results.
        exclude_studies : list of str, optional
            Studies to exclude from results. Cannot overlap with studies.
        
        Returns
        -------
        pd.DataFrame
            For single list input: DataFrame with analysis results.
        
        tuple (dict, dict)
            For DataFrame or dict input:
            - results_dict: {cluster_name: result_DataFrame}
            - top_hits_dict: {cluster_name: "predicted_cell_type" or "Unassigned"}
        
        Raises
        ------
        ValidationError
            If studies parameter contains invalid study names, or if
            studies and exclude_studies have overlapping values.
        
        Examples
        --------
        Single gene list (mouse cortex L6 markers):
        
        >>> df = client.analyzeGenes(
        ...     ["Syt6", "Tle4", "Hs3st4", "Fezf2", "Foxp2"],
        ...     species="Mouse"
        ... )
        
        With specific study filter:
        
        >>> results, top_hits = client.analyzeGenes(
        ...     cosg_marker_df,
        ...     n_top_genes=50,
        ...     min_genes=5,
        ...     studies=['AllenWholeMouseBrain2023_isocortex'],
        ...     species="Mouse"
        ... )
        
        Dictionary input:
        
        >>> gene_sets = {
        ...     'Microglia': ['Cx3cr1', 'P2ry12', 'Tmem119', 'Csf1r', 'Trem2'],
        ...     'L6_CT': ['Syt6', 'Tle4', 'Foxp2', 'Fezf2'],
        ... }
        >>> results, top_hits = client.analyzeGenes(gene_sets)
        """
        # Validate and normalize studies parameter
        studies_list = None
        if studies is not None:
            if isinstance(studies, str):
                studies_list = [studies]
            else:
                studies_list = list(studies)
            # a renamed study's former name is resolved to its current name (studyAliases)
            studies_list = self._resolve_studies(studies_list)
            
            # Validate study names against database
            available_studies = self.listStudies(species=species)
            invalid_studies = [s for s in studies_list if s not in available_studies]
            if invalid_studies:
                raise ValidationError(
                    f"Study '{invalid_studies[0]}' not found in PIASOmarkerDB. "
                    f"Use piaso.tl.queryPIASOmarkerDB(list_studies=True) to see available studies."
                )
        
        if exclude_studies is not None:
            exclude_studies = self._resolve_studies(
                [exclude_studies] if isinstance(exclude_studies, str) else list(exclude_studies))

        # Check for conflicts between studies and exclude_studies
        if studies_list is not None and exclude_studies is not None:
            overlap = set(studies_list) & set(exclude_studies)
            if overlap:
                raise ValidationError(
                    f"Conflicting parameters: '{list(overlap)[0]}' appears in both "
                    f"'studies' and 'exclude_studies'. These parameters cannot overlap."
                )
        
        # Handle DataFrame input (e.g., COSG output)
        if HAS_PANDAS and isinstance(genes, pd.DataFrame):
            gene_sets = {}
            for col in genes.columns:
                # Get top N genes, filter out None/NaN
                col_genes = genes[col].dropna().astype(str).tolist()[:n_top_genes]
                col_genes = [g for g in col_genes if g and g.strip() and g != 'nan']
                if col_genes:
                    gene_sets[str(col)] = col_genes
            
            return self._analyzeMultipleGeneSets(
                gene_sets,
                species=species,
                tissue=tissue,
                studies=studies_list,
                min_genes=min_genes,
                exclude_cell_types=exclude_cell_types,
                exclude_studies=exclude_studies,
            )
        
        # Handle dictionary input
        if isinstance(genes, dict):
            # Clean gene lists
            gene_sets = {}
            for name, gene_list in genes.items():
                if HAS_PANDAS:
                    cleaned = [
                        str(g) for g in gene_list 
                        if g is not None and pd.notna(g) and str(g).strip()
                    ][:n_top_genes]
                else:
                    cleaned = [
                        str(g) for g in gene_list 
                        if g is not None and str(g).strip()
                    ][:n_top_genes]
                if cleaned:
                    gene_sets[str(name)] = cleaned
            
            return self._analyzeMultipleGeneSets(
                gene_sets,
                species=species,
                tissue=tissue,
                studies=studies_list,
                min_genes=min_genes,
                exclude_cell_types=exclude_cell_types,
                exclude_studies=exclude_studies,
            )
        
        # Handle single list input
        if not genes:
            raise ValidationError("genes list cannot be empty")
        
        # Clean gene list
        if HAS_PANDAS:
            cleaned_genes = [
                str(g) for g in genes 
                if g is not None and pd.notna(g) and str(g).strip()
            ]
        else:
            cleaned_genes = [
                str(g) for g in genes 
                if g is not None and str(g).strip()
            ]
        
        result = self._analyzeSingleGeneList(
            genes=cleaned_genes,
            species=species,
            tissue=tissue,
            studies=studies_list,
            min_genes=min_genes,
            exclude_cell_types=exclude_cell_types,
            exclude_studies=exclude_studies
        )
        
        return self._resultsToDataFrame(result)
    
    def _marker_rows_for_genes(self, genes: List[str], species: str = None, tissue: str = None):
        """Every marker row of ``genes`` (with the species and tissue filters), asked for ``_GENE_BATCH``
        genes at a time so each URL stays short, as a DataFrame sorted by specificity score (descending,
        stable): the order one query for these genes returns."""
        genes = list(dict.fromkeys(genes))
        frames = []
        for i in range(0, len(genes), _GENE_BATCH):
            df = self.getAllMarkers(gene=genes[i:i + _GENE_BATCH], species=species, tissue=tissue,
                                    as_dict=False)
            if len(df):
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        if len(frames) > 1 and 'specificity_score' in df.columns:
            df = df.sort_values('specificity_score', ascending=False, kind='mergesort').reset_index(drop=True)
        return df

    def _analyzeSingleGeneList(
        self,
        genes: List[str],
        species: str = None,
        tissue: str = None,
        studies: List[str] = None,
        min_genes: int = 1,
        exclude_cell_types: List[str] = None,
        exclude_studies: List[str] = None,
    ) -> List[Dict]:
        """Internal method to analyze a single gene list."""
        if not genes:
            return []
        df = self._marker_rows_for_genes(genes, species=species, tissue=tissue)
        return self._analyze_rows(df, studies=studies, min_genes=min_genes,
                                  exclude_cell_types=exclude_cell_types, exclude_studies=exclude_studies)

    def _analyze_rows(
        self,
        df: 'pd.DataFrame',
        studies: List[str] = None,
        min_genes: int = 1,
        exclude_cell_types: List[str] = None,
        exclude_studies: List[str] = None,
    ) -> List[Dict]:
        """The cell types a gene list's marker rows point to: rows grouped by context (cell type, study,
        species, tissue, condition), each with its matched genes and their mean specificity, ranked by
        matched-gene count, then mean specificity."""
        if len(df) == 0:
            return []
        
        # Convert to list of dicts for processing
        all_matches = df.to_dict('records')
        
        # Filter by studies (inclusive filter) - apply BEFORE grouping
        if studies:
            all_matches = [
                m for m in all_matches 
                if m.get('study_publication', '') in studies
            ]
        
        if not all_matches:
            return []
        
        # Group by cell type context
        contexts = defaultdict(lambda: {
            'genes': [],
            'scores': [],
            'study': None,
            'species': None,
            'tissue': None,
            'condition': None
        })
        
        for m in all_matches:
            cell_type = m.get('cell_type', '')
            study_pub = m.get('study_publication', '')
            species_val = m.get('species', '')
            tissue_val = m.get('tissue', '')
            condition_val = m.get('condition', '')
            
            key = (cell_type, study_pub, species_val, tissue_val, condition_val)
            
            # Apply exclusion filters
            if exclude_cell_types and cell_type in exclude_cell_types:
                continue
            if exclude_studies and study_pub in exclude_studies:
                continue
            
            gene_name = m.get('gene', '')
            score = m.get('specificity_score', 0)
            
            contexts[key]['genes'].append(gene_name)
            contexts[key]['scores'].append(score)
            contexts[key]['study'] = study_pub
            contexts[key]['species'] = species_val
            contexts[key]['tissue'] = tissue_val
            contexts[key]['condition'] = condition_val
        
        # Build results
        results = []
        for (cell_type, study, species_val, tissue_val, condition), data in contexts.items():
            if len(data['genes']) >= min_genes:
                avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
                results.append({
                    'cell_type': cell_type,
                    'study_publication': study,
                    'species': species_val,
                    'tissue': tissue_val,
                    'condition': condition,
                    'matched_gene_count': len(data['genes']),
                    'matched_genes': data['genes'],
                    'avg_specificity': avg_score,
                    'specificity_scores': data['scores']
                })
        
        # Sort by gene count (desc), then by avg specificity (desc)
        results.sort(key=lambda x: (-x['matched_gene_count'], -x['avg_specificity']))
        
        return results
    
    def _analyzeMultipleGeneSets(
        self,
        gene_sets: Dict[str, List[str]],
        species: str = None,
        tissue: str = None,
        studies: List[str] = None,
        min_genes: int = 1,
        exclude_cell_types: List[str] = None,
        exclude_studies: List[str] = None,
    ) -> Tuple[Dict[str, 'pd.DataFrame'], Dict[str, str]]:
        """Analyze multiple gene sets and return results with top hits.

        The marker rows of every set's genes are asked for together (``_GENE_BATCH`` genes a request) and
        each set is analysed on its own genes' rows, the rows one query per set returned: a COSG table of
        30 clusters took 30 or more requests, now one per 200 distinct genes. The server matches gene
        symbols in any case, and so does the split here."""
        union = [g for genes in gene_sets.values() for g in genes]
        rows = self._marker_rows_for_genes(union, species=species, tissue=tissue) if union else pd.DataFrame()
        upper = rows['gene'].astype(str).str.upper() if len(rows) and 'gene' in rows.columns else None

        results_dict = {}
        top_hits_dict = {}
        for set_name, genes_list in gene_sets.items():
            if not genes_list or upper is None:
                results_dict[set_name] = pd.DataFrame()
                top_hits_dict[set_name] = self.NO_MATCH_VALUE
                continue
            mine = rows[upper.isin({str(g).upper() for g in genes_list}).to_numpy()]
            result = self._analyze_rows(mine, studies=studies, min_genes=min_genes,
                                        exclude_cell_types=exclude_cell_types,
                                        exclude_studies=exclude_studies)
            results_dict[set_name] = self._resultsToDataFrame(result)
            top_hits_dict[set_name] = result[0]['cell_type'] if result else self.NO_MATCH_VALUE

        return results_dict, top_hits_dict
    
    def _resultsToDataFrame(self, results: List[Dict]) -> 'pd.DataFrame':
        """Convert analysis results to DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required")
        
        if not results:
            return pd.DataFrame()
        
        df_data = []
        for r in results:
            df_data.append({
                'cell_type': r['cell_type'],
                'study_publication': r['study_publication'],
                'species': r['species'],
                'tissue': r['tissue'],
                'condition': r['condition'],
                'matched_gene_count': r['matched_gene_count'],
                'matched_genes': ','.join(r['matched_genes']),
                'avg_specificity': r['avg_specificity']
            })
        
        return pd.DataFrame(df_data)
    
    def _to_marker_dict(
        self,
        df: 'pd.DataFrame',
        gene_col: str = None,
        celltype_col: str = None,
    ) -> Dict[str, List[str]]:
        """
        Convert marker DataFrame to dictionary format.
        
        Returns
        -------
        dict
            Dictionary mapping cell types to lists of gene symbols.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required")
        
        if len(df) == 0:
            return {}
        
        # Auto-detect column names
        if gene_col is None:
            for col in ['gene', 'Gene', 'GENE', 'gene_symbol']:
                if col in df.columns:
                    gene_col = col
                    break
            else:
                gene_col = 'gene'
        
        if celltype_col is None:
            for col in ['cell_type', 'Cell_Type', 'celltype', 'CellType']:
                if col in df.columns:
                    celltype_col = col
                    break
            else:
                celltype_col = 'cell_type'
        
        if gene_col not in df.columns or celltype_col not in df.columns:
            return {}

        # One list per cell-type name. A name that several contexts (studies, tissues, conditions) use gets
        # the union of their markers, now said in a warning; a gene in several of them is listed once (it
        # was listed for each, and score() then counted it twice).
        ctx = [c for c in ("study_publication", "species", "tissue", "condition") if c in df.columns]
        if ctx:
            n_ctx = df[[celltype_col] + ctx].astype(str).drop_duplicates().groupby(celltype_col).size()
            merged = n_ctx[n_ctx > 1]
            if len(merged):
                warnings.warn(
                    f"PIASOmarkerDB: {len(merged)} cell type name(s) occur in several contexts (study, tissue or "
                    f"condition), each given one list, the union of its contexts' markers: "
                    f"{', '.join(map(str, merged.index[:10]))}{' ...' if len(merged) > 10 else ''}. Query one "
                    f"study and condition at a time to keep them apart.", UserWarning, stacklevel=3)
        marker_dict = {}
        for cell_type in df[celltype_col].unique():
            genes = df.loc[df[celltype_col] == cell_type, gene_col].tolist()
            marker_dict[cell_type] = list(dict.fromkeys(genes))

        return marker_dict
    
    # =========================================================================
    # Marker regions
    # =========================================================================

    def listRegionStudies(self, species: str = None) -> 'pd.DataFrame':
        """
        The studies whose marker regions PIASOmarkerDB holds.

        Parameters
        ----------
        species : str, optional
            ``'Human'`` or ``'Mouse'``, or the build, ``'hg38'`` or ``'mm10'``.

        Returns
        -------
        pd.DataFrame
            One row per study and context: ``study_publication``, ``species``, ``genome``, ``tissue``,
            ``condition``, ``cell_types`` and ``regions``.

        Examples
        --------
        >>> client.listRegionStudies(species="Mouse")
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for PIASOmarkerDB. Install with: pip install pandas")
        params = {}
        build = _region_genome(species, "species")
        if build:
            params['species'] = build
        data = self._extract_data(self._regions_request('region-studies', params))
        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame(columns=_REGION_STUDY_COLUMNS)
        return df[[c for c in _REGION_STUDY_COLUMNS if c in df.columns]
                  + [c for c in df.columns if c not in _REGION_STUDY_COLUMNS]]

    def _regions_request(self, endpoint: str, params: dict) -> Union[dict, list]:
        """A request to a marker-region endpoint; a server without them says so plainly."""
        try:
            return self._request('GET', endpoint, params=params)
        except APIError as e:
            if e.status_code == 404:
                raise APIError(
                    f"{self.base_url} has no marker regions (/api/{self.API_VERSION}/{endpoint} was not found); "
                    "they need a PIASOmarkerDB server from 2026-10 or later.", status_code=404) from e
            raise

    def getMarkerRegions(
        self,
        study: Union[str, List[str]] = None,
        cell_type: Union[str, List[str]] = None,
        species: str = None,
        genome: str = None,
        top_n: int = None,
        tissue: str = None,
        condition: str = None,
        gene: Union[str, List[str]] = None,
        region: Union[str, List[str]] = None,
        ccre: str = None,
        min_score: float = None,
        tile_size: int = 500,
        as_dict: bool = False,
        batch_size: int = 50_000,
    ) -> Union['pd.DataFrame', Dict[str, List[str]]]:
        """
        Query marker regions: the cell type-specific 500 bp tiles of single-nucleus chromatin atlases.

        Each region study ranks the 500 bp tiles of its atlas by COSG for every cell type and keeps up to 1,000
        per type, on one genome build (hg38 for human, mm10 for mouse). With ``as_dict=True`` the regions come
        back named as the tiles of a PIASO store, ``{cell_type: ['chr1:3000500-3001000', ...]}``, so the dict
        goes straight to ``kamon`` on a store's tile counts.

        Parameters
        ----------
        study : str or list of str, optional
            Region study or studies (see ``listRegionStudies``); a former study name works too.
        cell_type : str or list of str, optional
            Cell type(s) to keep.
        species : str, optional
            ``'Human'`` or ``'Mouse'``. Needed only when ``study`` and ``genome`` are both unset.
        genome : str, optional
            The genome build of your data, ``'hg38'`` or ``'mm10'``. A study on another build then raises an
            error instead of matching nothing: the regions are not lifted over between builds or species.
        top_n : int, optional
            The top ``top_n`` regions of each cell type, by COSG rank. Default: all of them (up to 1,000).
        tissue : str, optional
            Tissue to filter by.
        condition : str, optional
            Condition to filter by.
        gene : str or list of str, optional
            Keep the regions whose nearest gene, or one of whose nearby genes, is one of these.
        region : str or list of str, optional
            Keep the regions that overlap these intervals, ``'chr1:1000000-2000000'``.
        ccre : str, optional
            Keep the regions in a candidate cis-regulatory element of this class (``'dELS'``, ``'pELS'``,
            ``'PLS'``, ...).
        min_score : float, optional
            Minimum COSG score (non-negative).
        tile_size : int, optional
            The tile size, in bp, of the store whose tiles the IDs should name. Default: 500, PIASO's.
        as_dict : bool, optional
            ``True`` returns ``{cell_type: [tile IDs]}``, each list in rank order. ``False`` (the default)
            returns the DataFrame.
        batch_size : int, optional
            Rows per request, at most 100,000. Default: 50,000.

        Returns
        -------
        pd.DataFrame
            One row per region (``as_dict=False``), sorted by study, cell type and rank, with the server's
            fields: ``cell_type``, ``region``, ``chrom``, ``start`` and ``end`` (0-based, half-open),
            ``cosg_score``, ``cosg_rank``, ``study_publication``, ``species``, ``tissue``, ``condition``,
            ``genome``, ``ccre_class``, ``ccre_accession``, ``nearest_gene``, ``tss_distance``,
            ``nearby_genes`` and the conservation fields.

        dict
            ``{cell_type: [tile IDs]}`` (``as_dict=True``). A cell type of the same name in several studies
            gets one list; query one study at a time to keep them apart.

        Examples
        --------
        >>> regions = client.getMarkerRegions(
        ...     study="ZuRen2023_WholeBrain_ATAC", genome="mm10", top_n=200, as_dict=True
        ... )
        >>> df = client.getMarkerRegions(species="Human", gene="GAD2", ccre="dELS")
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for PIASOmarkerDB. Install with: pip install pandas")
        if as_dict is not True and as_dict is not False:
            raise ValueError(f"as_dict={as_dict!r} is not valid: pass True for the region dictionary or False "
                             "for the DataFrame.")
        top_n = _positive_int("top_n", top_n)
        tile_size = _positive_int("tile_size", tile_size)
        batch_size = min(_positive_int("batch_size", batch_size), _REGION_PAGE_MAX)
        if min_score is not None and min_score < 0:
            raise ValidationError("min_score must be non-negative")

        # the build: from species or genome, and checked against the studies' own
        want = _region_genome(genome, "genome")
        from_species = _region_genome(species, "species")
        if want and from_species and want != from_species:
            raise ValidationError(f"species={species!r} has its marker regions on {from_species}, not {genome!r}.")
        want = want or from_species
        studies = None
        if study is not None:
            names = [study] if isinstance(study, str) else list(study)
            names = [s.strip() for x in names for s in str(x).split(',') if s.strip()]
            studies = self._resolve_studies(names)
            known = self.listRegionStudies()
            builds = dict(zip(known['study_publication'], known['genome'])) if len(known) else {}
            folded = {s.lower(): s for s in builds}
            studies = [s if s in builds else folded.get(s.lower(), s) for s in studies]
            unknown = [s for s in studies if s not in builds]
            if unknown:
                raise ValidationError(
                    f"PIASOmarkerDB holds no marker regions for {', '.join(map(repr, unknown))}. The region studies "
                    f"are: {', '.join(sorted(builds)) or 'none'}. For marker genes, use getMarkers.")
            on = sorted({builds[s] for s in studies})
            if want and on != [want]:
                wrong = ', '.join(f"{s} ({builds[s]})" for s in studies if builds[s] != want)
                raise ValidationError(
                    f"genome {want} was asked for, but these studies' marker regions are on another build: {wrong}. "
                    "Marker regions are not lifted over between builds or species; choose a study on "
                    f"{want} (listRegionStudies).")
            if len(on) > 1:
                raise ValidationError(f"The studies are on {' and '.join(on)}; query one genome build at a time.")
            want = on[0]
        if not want:
            raise ValidationError("Pass species ('Human' or 'Mouse'), genome ('hg38' or 'mm10') or study: "
                                  "marker regions are kept per genome build.")

        base = {'species': want, 'sort': 'rank', 'order': 'asc'}
        if studies:
            base['study'] = ','.join(studies)
        if tissue:
            base['tissue'] = tissue
        if condition:
            base['condition'] = condition
        if gene:
            base['gene'] = gene if isinstance(gene, str) else ','.join(gene)
        if region:
            base['region'] = region if isinstance(region, str) else ';'.join(region)
        if ccre:
            base['ccre'] = ccre
        if min_score is not None:
            base['min_score'] = min_score
        # the server matches one cell type whole even with a comma in its name, and otherwise splits on commas;
        # a list holding such a name is asked for one cell type at a time
        if cell_type is None:
            groups = [None]
        elif isinstance(cell_type, str):
            groups = [cell_type]
        else:
            cts = [str(c) for c in cell_type]
            groups = cts if any(',' in c for c in cts) else [','.join(cts)]

        rows = []
        for ct in groups:
            params = dict(base)
            if ct is not None:
                params['cell_type'] = ct
            offset = 0
            while True:
                params.update(limit=batch_size, offset=offset)
                response = self._regions_request('regions', params)
                if isinstance(response, dict) and response.get('genome') not in (None, want):
                    raise APIError(f"{want} regions were asked for, and the server answered with "
                                   f"{response.get('genome')} regions.")
                page = self._extract_data(response)
                rows += page
                if not page:
                    break
                # pages come in rank order, so the top_n of every type is in once a page passes rank top_n
                by_rank = isinstance(response, dict) and response.get('sort') == 'rank'
                if by_rank and top_n is not None and (page[-1].get('cosg_rank') or 0) > top_n:
                    break
                has_more = response.get('has_more') if isinstance(response, dict) else None
                if has_more is False or (has_more is None and len(page) < batch_size):
                    break
                offset += len(page)

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=_REGION_COLUMNS)
        else:
            if 'region' not in df.columns and {'chrom', 'start', 'end'} <= set(df.columns):
                df['region'] = df['chrom'].astype(str) + ':' + df['start'].astype(str) + '-' + df['end'].astype(str)
            df = df[[c for c in _REGION_COLUMNS if c in df.columns]
                    + [c for c in df.columns if c not in _REGION_COLUMNS]]
            if top_n is not None:
                df = df[df['cosg_rank'] <= top_n]
            order = [c for c in ('study_publication', 'cell_type', 'cosg_rank', 'chrom', 'start') if c in df.columns]
            df = df.sort_values(order, kind='mergesort').reset_index(drop=True)
        if not as_dict:
            return df
        out: Dict[str, List[str]] = {}
        for ct, chrom, start, end in zip(df['cell_type'], df['chrom'], df['start'], df['end']):
            out.setdefault(ct, []).extend(_tile_ids(chrom, int(start), int(end), tile_size))
        return {ct: list(dict.fromkeys(ids)) for ct, ids in out.items()}

    # =========================================================================
    # Download
    # =========================================================================
    
    def downloadMarkers(
        self,
        filepath: str | Path,
        **kwargs
    ) -> None:
        """
        Download markers to CSV file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save the CSV file.
        **kwargs
            Same filter parameters as getMarkers() (species, tissue, study, etc.).
        
        Examples
        --------
        >>> client = PIASOmarkerDB()
        >>> client.downloadMarkers("human_markers.csv", species="Human")
        >>> client.downloadMarkers("mouse_brain.csv", species="Mouse", tissue="brain")
        """
        params = dict(kwargs)
        params['format'] = 'csv'
        
        url = self._build_url('markers')
        
        try:
            response = self._session.get(
                url, 
                params=params, 
                timeout=self.timeout * 2
            )
            
            response.raise_for_status()
            
            filepath = Path(filepath)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"Downloaded to {filepath}")
            
        except requests.exceptions.RequestException as e:
            raise APIError(f"Download failed: {e}")
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    @staticmethod
    def loadCustomMarkers(
        path: str | Path,
        gene_col: str = "Gene",
        celltype_col: str = "Cell_Type",
    ) -> Dict[str, List[str]]:
        """
        Load custom marker genes from a file.
        
        Parameters
        ----------
        path : str or Path
            Path to marker file. Supports CSV, JSON, pickle, and Excel.
        gene_col : str, optional
            Column name for gene symbols. Default: "Gene".
        celltype_col : str, optional
            Column name for cell types. Default: "Cell_Type".
        
        Returns
        -------
        dict
            Dictionary mapping cell types to lists of gene symbols.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required")
        
        path = Path(path)
        
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix == ".json":
            df = pd.read_json(path)
        elif path.suffix in [".pkl", ".pickle"]:
            df = pd.read_pickle(path)
        elif path.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")
        
        marker_dict = {}
        for cell_type in df[celltype_col].unique():
            genes = df[df[celltype_col] == cell_type][gene_col].tolist()
            marker_dict[cell_type] = list(dict.fromkeys(genes))      # a repeated gene once

        return marker_dict
    
    def getRecommendedStudy(
        self,
        species: str,
        tissue: str,
    ) -> str | None:
        """
        Get recommended study for a species/tissue combination.
        
        Parameters
        ----------
        species : {'human', 'mouse'}
            Species name.
        tissue : str
            Tissue type.
        
        Returns
        -------
        str or None
            Recommended study name, or None if no recommendation.
        
        Examples
        --------
        >>> client.getRecommendedStudy("human", "blood")
        'AllenHumanImmuneHealthAtlas2025_L2'
        >>> client.getRecommendedStudy("mouse", "cortex")
        'AllenWholeMouseBrain2023_isocortex'
        """
        recommendations = {
            ("human", "blood"): "AllenHumanImmuneHealthAtlas2025_L2",
            ("human", "pbmc"): "AllenHumanImmuneHealthAtlas2025_L2",
            ("human", "immune"): "AllenHumanImmuneHealthAtlas2025_L2",
            ("human", "brain"): "SilettiLinnarsson2023_WholeBrain_class",
            ("human", "cortex"): "SEAAD2024_MTG_subclass",
            ("human", "hippocampus"): "XuTeichmann2023_HippocampalFormation",
            ("human", "heart"): "XuTeichmann2023_Heart",
            ("human", "lung"): "XuTeichmann2023_Lung",
            ("human", "liver"): "XuTeichmann2023_Liver",
            ("human", "kidney"): "XuTeichmann2023_Kidney",
            ("human", "pancreas"): "XuTeichmann2023_Pancreas",
            ("human", "spleen"): "XuTeichmann2023_Spleen",
            ("human", "intestine"): "XuTeichmann2023_Intestine",
            ("mouse", "cortex"): "AllenWholeMouseBrain2023_isocortex",
            ("mouse", "brain"): "AllenWholeMouseBrain2023_Neuron",
        }
        
        key = (species.lower(), tissue.lower())
        return recommendations.get(key)


# =============================================================================
# Module-level Default Client
# =============================================================================

_default_client: PIASOmarkerDB | None = None


def _get_client() -> PIASOmarkerDB:
    """Get or create default client instance."""
    global _default_client
    if _default_client is None:
        _default_client = PIASOmarkerDB()
    return _default_client


# =============================================================================
# Main Public Functions
# =============================================================================

def queryPIASOmarkerDB(
    gene: Union[str, List[str]] = None,
    cell_type: Union[str, List[str]] = None,
    study: str = None,
    species: str = None,
    tissue: str = None,
    condition: str = None,
    min_score: float = None,
    max_score: float = None,
    limit: int = None,
    as_dict: Union[bool, str] = False,
    list_studies: bool = False,
    list_cell_types: bool = False,
    list_genes: bool = False,
) -> Union['pd.DataFrame', Tuple['pd.DataFrame', Dict], List[str]]:
    """
    Query PIASOmarkerDB for cell type marker genes.
    
    This is the main entry point for accessing PIASOmarkerDB through PIASO.
    
    Parameters
    ----------
    gene : str or list of str, optional
        Gene symbol(s) to filter by.
    cell_type : str or list of str, optional
        Cell type(s) to filter by.
    study : str, optional
        Study/publication to filter by.
    species : str, optional
        Species to filter by (e.g., "Human", "Mouse").
    tissue : str, optional
        Tissue to filter by.
    condition : str, optional
        Condition to filter by.
    min_score : float, optional
        Minimum specificity score (>= 0).
    max_score : float, optional
        Maximum specificity score (>= 0).
    limit : int, optional
        Maximum results to return. Default: None (no limit).
    as_dict : bool or {'both'}, optional
        ``True`` returns the ``{cell_type: [genes]}`` dictionary alone — the
        form ``kamon`` takes. ``False`` (the default) returns
        the DataFrame. ``'both'`` returns the old ``(DataFrame, dict)`` pair
        and is deprecated; it will be removed in the next release.
    list_studies : bool, optional
        If True, return list of available study names instead of markers.
        Default: False.
    list_cell_types : bool, optional
        If True, return list of available cell types instead of markers.
        Default: False.
    list_genes : bool, optional
        If True, return list of unique gene symbols instead of markers.
        Default: False.
    
    Returns
    -------
    pd.DataFrame
        Marker query results (default).
    
    dict
        If as_dict=True: ``{cell_type: [genes]}``.

    tuple (pd.DataFrame, dict)
        If as_dict='both' (deprecated).

    list of str
        If list_studies=True, list_cell_types=True, or list_genes=True.
    
    Examples
    --------
    Query marker genes:
    
    >>> import piaso
    >>> df = piaso.tl.queryPIASOmarkerDB(gene="Foxp2", species="Mouse")
    >>> df = piaso.tl.queryPIASOmarkerDB(gene=["Foxp2", "Syt6", "Tle4"])
    
    Get the marker dictionary, ready for kamon:

    >>> marker_dict = piaso.tl.queryPIASOmarkerDB(
    ...     study="AllenWholeMouseBrain2023_isocortex",
    ...     species="Mouse",
    ...     as_dict=True
    ... )
    >>> print(f"Cell types in dict: {len(marker_dict)}")
    >>> piaso.tl.kamon(adata, marker_gene_set=marker_dict)
    
    List available studies:
    
    >>> studies = piaso.tl.queryPIASOmarkerDB(list_studies=True)
    >>> print(f"Total studies: {len(studies)}")
    
    List cell types:
    
    >>> cell_types = piaso.tl.queryPIASOmarkerDB(list_cell_types=True, species="Mouse")
    
    See Also
    --------
    analyzeMarkers : Analyze gene lists for cell type inference
    PIASOmarkerDB : Client class for advanced usage
    
    Notes
    -----
    PIASOmarkerDB website: https://piaso.org/piasomarkerdb/
    """
    client = _get_client()
    
    # Handle meta queries
    if list_studies:
        return client.listStudies(species=species, tissue=tissue)
    
    if list_cell_types:
        return client.getCellTypes(species=species, tissue=tissue, study=study)
    
    if list_genes:
        return client.getGenes(
            cell_type=cell_type,
            species=species,
            tissue=tissue,
            min_score=min_score
        )
    
    # Standard marker query
    return client.getMarkers(
        gene=gene,
        cell_type=cell_type,
        study=study,
        species=species,
        tissue=tissue,
        condition=condition,
        min_score=min_score,
        max_score=max_score,
        limit=limit,
        as_dict=as_dict,
    )


@wraps(queryPIASOmarkerDB)
def getMarkers(*args, **kwargs):
    """
    Alias for :func:`queryPIASOmarkerDB`.
    
    Please refer to the main function for full documentation.
    """
    return queryPIASOmarkerDB(*args, **kwargs)


def getMarkerRegions(
    study: Union[str, List[str]] = None,
    cell_type: Union[str, List[str]] = None,
    species: str = None,
    genome: str = None,
    top_n: int = None,
    tissue: str = None,
    condition: str = None,
    gene: Union[str, List[str]] = None,
    region: Union[str, List[str]] = None,
    ccre: str = None,
    min_score: float = None,
    tile_size: int = 500,
    as_dict: bool = False,
    list_studies: bool = False,
) -> Union['pd.DataFrame', Dict[str, List[str]]]:
    """
    Query PIASOmarkerDB for marker regions, the cell type-specific tiles of chromatin atlases.

    Each region study ranks the 500 bp tiles of a single-nucleus chromatin atlas by COSG for every cell type and
    keeps up to 1,000 per type, on one genome build (hg38 for human, mm10 for mouse). ``as_dict=True`` returns them
    named as the tiles of a PIASO store, ``{cell_type: ['chr1:3000500-3001000', ...]}``: the form
    ``kamon`` takes, here on a store's tile counts rather than on genes.

    Parameters
    ----------
    study : str or list of str, optional
        Region study or studies (``list_studies=True`` lists them); a former study name works too.
    cell_type : str or list of str, optional
        Cell type(s) to keep.
    species : str, optional
        ``'Human'`` or ``'Mouse'``. Needed only when ``study`` and ``genome`` are both unset.
    genome : str, optional
        The genome build of your data, ``'hg38'`` or ``'mm10'``. A study on another build then raises an error
        instead of matching nothing: regions are not lifted over between builds or species.
    top_n : int, optional
        The top ``top_n`` regions of each cell type, by COSG rank. Default: all of them (up to 1,000).
    tissue : str, optional
        Tissue to filter by.
    condition : str, optional
        Condition to filter by.
    gene : str or list of str, optional
        Keep the regions whose nearest gene, or one of whose nearby genes, is one of these.
    region : str or list of str, optional
        Keep the regions that overlap these intervals, ``'chr1:1000000-2000000'``.
    ccre : str, optional
        Keep the regions in a candidate cis-regulatory element of this class (``'dELS'``, ``'pELS'``, ``'PLS'``,
        ...).
    min_score : float, optional
        Minimum COSG score (non-negative).
    tile_size : int, optional
        The tile size, in bp, of the store whose tiles the IDs should name. Default: 500, PIASO's.
    as_dict : bool, optional
        ``True`` returns ``{cell_type: [tile IDs]}``, each list in rank order. ``False`` (the default) returns
        the DataFrame.
    list_studies : bool, optional
        If True, return the region studies (a DataFrame) instead of regions. Default: False.

    Returns
    -------
    pd.DataFrame
        One row per region (default), sorted by study, cell type and rank: the region (``chrom``, ``start`` and
        ``end``, 0-based and half-open), its cell type and study, the COSG score and rank, the genome build, the
        cCRE class, the nearest and nearby genes, and the conservation fields.

    dict
        ``{cell_type: [tile IDs]}`` (as_dict=True). A cell type of the same name in several studies gets one list;
        query one study at a time to keep them apart.

    pd.DataFrame
        The region studies (list_studies=True).

    Examples
    --------
    Name single nuclei from their tile counts, with the whole-brain atlas's regions:

    >>> import piaso
    >>> regions = piaso.tl.getMarkerRegions(
    ...     study="ZuRen2023_WholeBrain_ATAC", genome="mm10", top_n=200, as_dict=True
    ... )
    >>> # tiles: an AnnData of a PIASO store's 500 bp tile counts (var_names such as 'chr1:3000500-3001000'),
    >>> # normalized by piaso.tl.infog into layers['infog'], with the nuclei's embedding in obsm['X_svd']
    >>> piaso.tl.kamon(tiles, marker_gene_set=regions, use_rep="X_svd")

    The distal elements near a gene, in every human study:

    >>> df = piaso.tl.getMarkerRegions(species="Human", gene="GAD2", ccre="dELS")

    List the region studies:

    >>> piaso.tl.getMarkerRegions(list_studies=True)

    See Also
    --------
    getMarkers : Marker genes
    kamon : Name cells from a marker dict (also available as ``predictCellTypeByMarker``)

    Notes
    -----
    PIASOmarkerDB website: https://piaso.org/piasomarkerdb/
    """
    client = _get_client()
    if list_studies:
        return client.listRegionStudies(species=species or genome)
    return client.getMarkerRegions(
        study=study,
        cell_type=cell_type,
        species=species,
        genome=genome,
        top_n=top_n,
        tissue=tissue,
        condition=condition,
        gene=gene,
        region=region,
        ccre=ccre,
        min_score=min_score,
        tile_size=tile_size,
        as_dict=as_dict,
    )


def analyzeMarkers(
    genes: Union[List[str], 'pd.DataFrame', Dict[str, List[str]]],
    n_top_genes: int = 50,
    species: str = None,
    tissue: str = None,
    studies: Union[str, List[str]] = None,
    min_genes: int = 1,
    exclude_cell_types: List[str] = None,
    exclude_studies: List[str] = None,
) -> Union['pd.DataFrame', Tuple[Dict[str, 'pd.DataFrame'], Dict[str, str]]]:
    """
    Analyze gene list(s) to infer potential cell types.
    
    This function queries PIASOmarkerDB to find which cell types are associated
    with the input genes, ranking results by matched gene count and specificity.
    
    Parameters
    ----------
    genes : list of str, pd.DataFrame, or dict
        Gene input. Supports three formats:
        
        - **list of str**: Single list of gene symbols.
          Returns: pd.DataFrame with analysis results.
          
        - **pd.DataFrame**: Columns are clusters/cell types, rows are genes.
          Ideal for COSG output: ``pd.DataFrame(adata.uns['cosg']['names']).head(50)``.
          Returns: tuple (results_dict, top_hits_dict).
          
        - **dict**: ``{cluster_name: [gene_list]}``.
          Returns: tuple (results_dict, top_hits_dict).
          
    n_top_genes : int, optional
        For DataFrame/dict input: only use top N genes per column/key.
        Default: 50. Useful for COSG results which may rank many genes.
    species : str, optional
        Filter results by species (e.g., "Human", "Mouse").
    tissue : str, optional
        Filter results by tissue.
    studies : str or list of str, optional
        Study/studies to include in analysis. Only cell types from these
        studies will be considered. Study names are validated against
        PIASOmarkerDB; a renamed study's former name is resolved to its
        current name, with a warning (``PIASOmarkerDB.studyAliases``). Cannot overlap with exclude_studies.
        Default: None (use all studies).
    min_genes : int, optional
        Minimum number of genes that must match a cell type. Default: 1.
    exclude_cell_types : list of str, optional
        Cell types to exclude from results.
    exclude_studies : list of str, optional
        Studies to exclude from results. Cannot overlap with studies.
    
    Returns
    -------
    pd.DataFrame
        For single list input: DataFrame with columns:
        cell_type, study_publication, species, tissue, condition,
        matched_gene_count, matched_genes, avg_specificity.
    
    tuple (dict, dict)
        For DataFrame or dict input:
        
        - **results_dict**: ``{cluster_name: result_DataFrame}``
        - **top_hits_dict**: ``{cluster_name: "predicted_cell_type"}``
          (or "Unassigned" if no matches found)
    
    Raises
    ------
    ValidationError
        If studies parameter contains invalid study names, or if
        studies and exclude_studies have overlapping values.
    
    Examples
    --------
    Single gene list (mouse cortex L6 markers):
    
    >>> import piaso
    >>> query_genes = ["Syt6", "Tle4", "Hs3st4", "Fezf2", "Foxp2", "Col12a1"]
    >>> df = piaso.tl.analyzeMarkers(query_genes)
    >>> print(df.head())
    # Top hit: EN-L6-CT from WangKriegstein2025
    
    With specific study filter:
    
    >>> results, top_hits = piaso.tl.analyzeMarkers(
    ...     cosg_marker_df,
    ...     n_top_genes=50,
    ...     min_genes=5,
    ...     studies=['AllenWholeMouseBrain2023_isocortex'],
    ...     species="Mouse"
    ... )
    >>> print(top_hits)
    {'L2-3 IT': '007 L2/3 IT CTX Glut', 'PV': '052 Pvalb Gaba', ...}
    
    Dictionary input (microglia and L6 CT markers):
    
    >>> gene_sets = {
    ...     'Cluster_0': ['Cx3cr1', 'P2ry12', 'Tmem119', 'Csf1r', 'Trem2'],
    ...     'Cluster_1': ['Syt6', 'Tle4', 'Hs3st4', 'Fezf2', 'Foxp2'],
    ... }
    >>> results, top_hits = piaso.tl.analyzeMarkers(gene_sets)
    >>> print(top_hits)
    {'Cluster_0': 'Microglia', 'Cluster_1': 'EN-L6-CT'}
    
    COSG integration workflow:
    
    >>> import cosg
    >>> import pandas as pd
    >>> 
    >>> # Run COSG
    >>> cosg.cosg(adata, key_added='cosg', groupby='leiden')
    >>> 
    >>> # Get top 50 markers per cluster
    >>> cosg_marker_df = pd.DataFrame(adata.uns['cosg']['names']).head(50)
    >>> 
    >>> # Analyze with PIASOmarkerDB
    >>> results, top_hits = piaso.tl.analyzeMarkers(
    ...     cosg_marker_df,
    ...     n_top_genes=50,
    ...     species="Mouse"
    ... )
    >>> 
    >>> # Add annotations to AnnData
    >>> adata.obs['cell_type_predicted'] = adata.obs['leiden'].map(top_hits)
    
    See Also
    --------
    queryPIASOmarkerDB : Direct marker queries
    
    Notes
    -----
    For COSG results, the DataFrame columns are cluster/cell type names and
    rows contain the ranked marker genes. Only the top ``n_top_genes`` genes
    per column are used for analysis.
    
    When ``studies`` is provided, study names are validated against PIASOmarkerDB.
    If an invalid study name is provided, a ValidationError is raised with
    instructions to list available studies.
    
    PIASOmarkerDB website: https://piaso.org/piasomarkerdb/
    """
    client = _get_client()
    return client.analyzeGenes(
        genes,
        n_top_genes=n_top_genes,
        species=species,
        tissue=tissue,
        studies=studies,
        min_genes=min_genes,
        exclude_cell_types=exclude_cell_types,
        exclude_studies=exclude_studies,
    )


# =============================================================================
# Module Metadata
# =============================================================================

__all__ = [
    'queryPIASOmarkerDB',
    'getMarkers',
    'analyzeMarkers',
    'PIASOmarkerDB',
]

__version__ = "1.0.3"