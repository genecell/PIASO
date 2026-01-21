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
    df = piaso.tl.queryPIASOmarkerDB(gene="CD3E", species="Human")
    
    # Get both DataFrame and marker dict
    df, marker_dict = piaso.tl.queryPIASOmarkerDB(
        study="AllenHumanImmuneHealthAtlas_L2",
        species="Human", 
        as_dict=True
    )
    
    # List available studies
    studies = piaso.tl.queryPIASOmarkerDB(list_studies=True)
    
    # Analyze gene lists for cell type inference
    df = piaso.tl.analyzeMarkers(["CD3E", "CD8A", "GZMK"])
    
    # Analyze COSG results (DataFrame input) with specific study
    import pandas as pd
    cosg_df = pd.DataFrame(adata.uns['cosg']['names'])
    results, top_hits = piaso.tl.analyzeMarkers(
        cosg_df, 
        n_top_genes=50,
        studies="SEAAD2024_MTG_Subclass"
    )
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# =============================================================================
# Exceptions (Internal Use Only - Not Exported)
# =============================================================================

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
    """Authentication/invitation code error for download access."""
    pass


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
    invitation_code : str, optional
        Invitation code for bulk download access.
        Contact dai@broadinstitute.org to request.
    timeout : int, optional
        Request timeout in seconds. Default: 30
    cache_dir : str or Path, optional
        Directory for caching downloaded markers.
        Default: ~/.piaso/markers
    
    Examples
    --------
    Basic usage:
    
    >>> from piaso.tools import PIASOmarkerDB
    >>> client = PIASOmarkerDB()
    >>> 
    >>> # Query markers
    >>> df = client.getMarkers(gene="CD3E")
    >>> 
    >>> # Get as dict
    >>> df, marker_dict = client.getMarkers(gene="CD3E", as_dict=True)
    
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
    
    # Return value for no matches
    NO_MATCH_VALUE = "Unassigned"
    
    def __init__(
        self,
        base_url: str = None,
        invitation_code: str = None,
        timeout: int = None,
        cache_dir: str | Path | None = None,
    ):
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip('/')
        self.invitation_code = invitation_code
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self._session = requests.Session()
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".piaso" / "markers"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        try:
            response = self._session.request(
                method=method,
                url=url,
                params=params,
                timeout=self.timeout,
                **kwargs
            )
            
            if response.status_code == 403:
                raise AuthenticationError(
                    "Invalid or missing invitation code"
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
            
            return response.json()
            
        except requests.exceptions.Timeout:
            raise MarkerDBConnectionError(
                f"Request timed out after {self.timeout}s"
            )
        except requests.exceptions.ConnectionError as e:
            raise MarkerDBConnectionError(
                f"Failed to connect to {url}: {e}"
            )
    
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
        as_dict: bool = False,
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
        as_dict : bool, optional
            If True, also return marker dictionary {cell_type: [genes]}.
            Default: False.
        
        Returns
        -------
        pd.DataFrame
            Query results (if as_dict=False)
        
        tuple (pd.DataFrame, dict)
            Query results and marker dictionary (if as_dict=True)
        
        Examples
        --------
        >>> df = client.getMarkers(gene="CD3E", species="Human")
        >>> df, marker_dict = client.getMarkers(study="SEAAD2024_MTG_Subclass", as_dict=True)
        """
        params = {}
        
        # Handle gene parameter
        if gene:
            if isinstance(gene, list):
                params['gene'] = ','.join(gene)
            else:
                params['gene'] = gene
        
        # Handle cell_type parameter
        if cell_type:
            if isinstance(cell_type, list):
                params['cell_type'] = ','.join(cell_type)
            else:
                params['cell_type'] = cell_type
        
        # Other filters
        if study:
            params['study'] = study
        if species:
            params['species'] = species
        if tissue:
            params['tissue'] = tissue
        if condition:
            params['condition'] = condition
        
        # Score filters with validation
        if min_score is not None:
            if min_score < 0:
                raise ValidationError("min_score must be non-negative")
            params['min_score'] = min_score
        
        if max_score is not None:
            if max_score < 0:
                raise ValidationError("max_score must be non-negative")
            params['max_score'] = max_score
        
        # Pagination
        if limit is not None:
            params['limit'] = limit
        if offset > 0:
            params['offset'] = offset
        
        # Make request
        response = self._request('GET', 'markers', params=params)
        data = self._extract_data(response)
        
        # Convert to DataFrame
        if not HAS_PANDAS:
            raise ImportError(
                "pandas is required for PIASOmarkerDB. "
                "Install with: pip install pandas"
            )
        
        df = pd.DataFrame(data)
        
        if as_dict:
            marker_dict = self._to_marker_dict(df)
            return df, marker_dict
        else:
            return df
    
    def getAllMarkers(
        self,
        batch_size: int = 10000,
        verbose: bool = False,
        as_dict: bool = False,
        **kwargs
    ) -> Union['pd.DataFrame', Tuple['pd.DataFrame', Dict[str, List[str]]]]:
        """
        Get all markers matching filters with automatic pagination.
        
        Parameters
        ----------
        batch_size : int, optional
            Number of records per request. Default: 10000.
        verbose : bool, optional
            Print progress information. Default: False.
        as_dict : bool, optional
            If True, also return marker dictionary. Default: False.
        **kwargs
            Same filter parameters as getMarkers().
        
        Returns
        -------
        pd.DataFrame or tuple
            All matching markers (and optionally marker dict).
        """
        all_data = []
        offset = 0
        batch_num = 0
        
        # Remove as_dict from kwargs for internal calls
        kwargs_copy = dict(kwargs)
        kwargs_copy.pop('as_dict', None)
        
        while True:
            batch_num += 1
            
            df_batch = self.getMarkers(
                limit=batch_size,
                offset=offset,
                as_dict=False,
                **kwargs_copy
            )
            
            if len(df_batch) == 0:
                break
            
            all_data.append(df_batch)
            
            if verbose:
                print(f"Fetching markers... batch {batch_num} ({len(df_batch)} records)")
            
            if len(df_batch) < batch_size:
                break
            
            offset += batch_size
        
        if not all_data:
            df = pd.DataFrame()
        else:
            df = pd.concat(all_data, ignore_index=True)
        
        if verbose:
            print(f"Total: {len(df)} markers")
        
        if as_dict:
            marker_dict = self._to_marker_dict(df)
            return df, marker_dict
        else:
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
              E.g., COSG output: pd.DataFrame(adata.uns['cosg']['names']).
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
            PIASOmarkerDB. Cannot overlap with exclude_studies.
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
        Single gene list:
        
        >>> df = client.analyzeGenes(["CD3E", "CD8A", "GZMK"])
        
        With specific study:
        
        >>> df = client.analyzeGenes(
        ...     ["CD3E", "CD8A", "GZMK"],
        ...     studies="SEAAD2024_MTG_Subclass"
        ... )
        
        COSG output (DataFrame) with multiple studies:
        
        >>> cosg_df = pd.DataFrame(adata.uns['cosg']['names'])
        >>> results, top_hits = client.analyzeGenes(
        ...     cosg_df, 
        ...     n_top_genes=50,
        ...     studies=["SEAAD2024_MTG_Subclass", "SilettiLinnarssonWholeHumanBrain2023_class"]
        ... )
        >>> print(top_hits)
        {'Lamp5': 'GABAergic neuron', 'Lhx6': 'Interneuron', ...}
        
        Dictionary input:
        
        >>> results, top_hits = client.analyzeGenes({
        ...     'Cluster_0': ['CD3E', 'CD8A', 'GZMK'],
        ...     'Cluster_1': ['MS4A1', 'CD19', 'CD79A'],
        ... })
        """
        # Validate and normalize studies parameter
        studies_list = None
        if studies is not None:
            if isinstance(studies, str):
                studies_list = [studies]
            else:
                studies_list = list(studies)
            
            # Validate study names against database
            available_studies = self.listStudies(species=species)
            invalid_studies = [s for s in studies_list if s not in available_studies]
            if invalid_studies:
                raise ValidationError(
                    f"Study '{invalid_studies[0]}' not found in PIASOmarkerDB. "
                    f"Use piaso.tl.queryPIASOmarkerDB(list_studies=True) to see available studies."
                )
        
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
        
        # Get all markers for the input genes
        df = self.getAllMarkers(
            gene=genes,
            species=species,
            tissue=tissue,
            as_dict=False
        )
        
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
        """Analyze multiple gene sets and return results with top hits."""
        results_dict = {}
        top_hits_dict = {}
        
        for set_name, genes_list in gene_sets.items():
            if not genes_list:
                results_dict[set_name] = pd.DataFrame()
                top_hits_dict[set_name] = self.NO_MATCH_VALUE
                continue
            
            # Analyze this gene set
            result = self._analyzeSingleGeneList(
                genes=genes_list,
                species=species,
                tissue=tissue,
                studies=studies,
                min_genes=min_genes,
                exclude_cell_types=exclude_cell_types,
                exclude_studies=exclude_studies
            )
            
            # Store results
            results_dict[set_name] = self._resultsToDataFrame(result)
            
            # Get top hit
            if result:
                top_hits_dict[set_name] = result[0]['cell_type']
            else:
                top_hits_dict[set_name] = self.NO_MATCH_VALUE
        
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
        
        marker_dict = {}
        for cell_type in df[celltype_col].unique():
            genes = df[df[celltype_col] == cell_type][gene_col].tolist()
            marker_dict[cell_type] = genes
        
        return marker_dict
    
    # =========================================================================
    # Download
    # =========================================================================
    
    def downloadMarkers(
        self,
        filepath: str | Path,
        invitation_code: str = None,
        **kwargs
    ) -> None:
        """
        Download markers to CSV file.
        
        Requires an invitation code. Contact dai@broadinstitute.org to request.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save the CSV file.
        invitation_code : str, optional
            Invitation code. Uses client default if not provided.
        **kwargs
            Same filter parameters as getMarkers().
        """
        code = invitation_code or self.invitation_code
        if not code:
            raise AuthenticationError(
                "Invitation code required for download. "
                "Request one from dai@broadinstitute.org"
            )
        
        params = dict(kwargs)
        params['invitation_code'] = code
        params['format'] = 'csv'
        
        url = self._build_url('markers')
        
        try:
            response = self._session.get(
                url, 
                params=params, 
                timeout=self.timeout * 2
            )
            
            if response.status_code == 403:
                raise AuthenticationError("Invalid invitation code")
            
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
            marker_dict[cell_type] = genes
        
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
        """
        recommendations = {
            ("human", "blood"): "AllenHumanImmuneHealthAtlas_L2",
            ("human", "pbmc"): "AllenHumanImmuneHealthAtlas_L2",
            ("human", "immune"): "AllenHumanImmuneHealthAtlas_L2",
            ("human", "brain"): "SilettiLinnarssonWholeHumanBrain2023_class",
            ("human", "cortex"): "SEAAD2024_MTG_Subclass",
            ("human", "hippocampus"): "XuTeichmann2023_HippocampalFormation",
            ("human", "heart"): "XuTeichmann2023_Heart",
            ("human", "lung"): "XuTeichmann2023_Lung",
            ("human", "liver"): "XuTeichmann2023_Liver",
            ("human", "kidney"): "XuTeichmann2023_Kidney",
            ("human", "pancreas"): "XuTeichmann2023_Pancreas",
            ("human", "spleen"): "XuTeichmann2023_Spleen",
            ("human", "intestine"): "XuTeichmann2023_Intestine",
            ("mouse", "cortex"): "AllenWholeMouseBrain_isocortex",
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
    as_dict: bool = False,
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
    as_dict : bool, optional
        If True, also return {cell_type: [genes]} dictionary.
        Returns tuple (DataFrame, dict). Default: False.
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
    
    tuple (pd.DataFrame, dict)
        If as_dict=True: (DataFrame, {cell_type: [genes]}).
    
    list of str
        If list_studies=True, list_cell_types=True, or list_genes=True.
    
    Examples
    --------
    Query marker genes:
    
    >>> import piaso
    >>> df = piaso.tl.queryPIASOmarkerDB(gene="CD3E", species="Human")
    
    Get both DataFrame and marker dictionary:
    
    >>> df, marker_dict = piaso.tl.queryPIASOmarkerDB(
    ...     study="SEAAD2024_MTG_Subclass",
    ...     species="Human",
    ...     as_dict=True
    ... )
    
    List available studies:
    
    >>> studies = piaso.tl.queryPIASOmarkerDB(list_studies=True)
    >>> studies = piaso.tl.queryPIASOmarkerDB(list_studies=True, species="Human")
    
    List cell types:
    
    >>> cell_types = piaso.tl.queryPIASOmarkerDB(list_cell_types=True, species="Human")
    
    List genes:
    
    >>> genes = piaso.tl.queryPIASOmarkerDB(list_genes=True, cell_type="T-cell")
    
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
          Ideal for COSG output: ``pd.DataFrame(adata.uns['cosg']['names'])``.
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
        PIASOmarkerDB. Cannot overlap with exclude_studies.
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
    Single gene list:
    
    >>> import piaso
    >>> df = piaso.tl.analyzeMarkers(["CD3E", "CD8A", "GZMK", "PRF1"])
    >>> print(df.head())
    
    With specific study filter:
    
    >>> df = piaso.tl.analyzeMarkers(
    ...     ["CD3E", "CD8A", "GZMK"],
    ...     studies="SEAAD2024_MTG_Subclass"
    ... )
    
    With multiple studies:
    
    >>> df = piaso.tl.analyzeMarkers(
    ...     ["CD3E", "CD8A", "GZMK"],
    ...     studies=["SEAAD2024_MTG_Subclass", "SilettiLinnarssonWholeHumanBrain2023_class"]
    ... )
    
    COSG output (DataFrame with columns as clusters):
    
    >>> import pandas as pd
    >>> cosg_df = pd.DataFrame(adata.uns['cosg']['names'])
    >>> results, top_hits = piaso.tl.analyzeMarkers(
    ...     cosg_df,
    ...     n_top_genes=50,
    ...     species="Human",
    ...     studies="SEAAD2024_MTG_Subclass"
    ... )
    >>> print(top_hits)
    {'Lamp5': 'Lamp5', 'Sst': 'Sst', 'Pvalb': 'Pvalb', ...}
    
    Dictionary input:
    
    >>> results, top_hits = piaso.tl.analyzeMarkers({
    ...     'Cluster_0': ['CD3E', 'CD8A', 'GZMK'],
    ...     'Cluster_1': ['MS4A1', 'CD19', 'CD79A'],
    ...     'Cluster_2': ['LYZ', 'CD14', 'FCGR3A'],
    ... }, species="Human")
    >>> print(top_hits)
    {'Cluster_0': 'CD8+ T cell', 'Cluster_1': 'B cell', 'Cluster_2': 'Monocyte'}
    
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
