"""
GIS Integration and Spatial Data Processing for NetLogo BEE-STEWARD v2 Parity
============================================================================

Advanced GIS capabilities for processing real-world spatial data, coordinate transformations,
raster/vector analysis, and integration with external GIS datasets.
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import logging
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import tempfile

# Optional GIS dependencies
try:
    import pyproj
    from pyproj import CRS, Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    logging.warning("pyproj not available - coordinate transformations will be limited")

try:
    import rasterio
    from rasterio.features import shapes, rasterize
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logging.warning("rasterio not available - raster processing will be limited")

try:
    import geopandas as gpd
    import shapely
    from shapely.geometry import Point, Polygon, LineString, MultiPolygon
    from shapely.ops import transform, unary_union
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    logging.warning("geopandas/shapely not available - vector processing will be limited")

from .spatial_algorithms import SpatialPoint, SpatialPatch, SpatialIndex

class CoordinateSystem(Enum):
    """Supported coordinate systems"""
    WGS84 = "EPSG:4326"  # GPS coordinates
    WEB_MERCATOR = "EPSG:3857"  # Web mapping standard
    UTM_AUTO = "UTM_AUTO"  # Automatic UTM zone selection
    LOCAL = "LOCAL"  # Local coordinate system

class DataFormat(Enum):
    """Supported spatial data formats"""
    GEOJSON = "geojson"
    SHAPEFILE = "shapefile"
    KML = "kml"
    GEOTIFF = "geotiff"
    CSV_POINTS = "csv_points"
    WKT = "wkt"
    GPKG = "geopackage"

class LandCoverType(Enum):
    """Land cover classification types"""
    URBAN = "urban"
    AGRICULTURAL = "agricultural"
    FOREST = "forest"
    GRASSLAND = "grassland"
    WATER = "water"
    WETLAND = "wetland"
    BARREN = "barren"
    DEVELOPED = "developed"
    MIXED = "mixed"

@dataclass
class GeoTransform:
    """Geospatial transformation parameters"""
    source_crs: str
    target_crs: str
    transform_matrix: Optional[np.ndarray] = None
    pixel_size: Optional[Tuple[float, float]] = None
    bounds: Optional[Tuple[float, float, float, float]] = None

@dataclass
class RasterMetadata:
    """Raster dataset metadata"""
    width: int
    height: int
    bands: int
    data_type: str
    crs: str
    transform: GeoTransform
    nodata_value: Optional[float] = None
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)

@dataclass
class VectorMetadata:
    """Vector dataset metadata"""
    geometry_type: str
    feature_count: int
    crs: str
    bounds: Tuple[float, float, float, float]
    fields: List[Dict[str, str]] = field(default_factory=list)
    spatial_index: bool = False

class CoordinateTransformer(BaseModel):
    """Handle coordinate system transformations"""
    
    model_config = {"validate_assignment": True}
    
    # Configuration
    source_crs: str = CoordinateSystem.WGS84.value
    target_crs: str = CoordinateSystem.LOCAL.value
    
    # Internal state
    _transformer: Optional[Any] = None
    _inverse_transformer: Optional[Any] = None
    
    # Logger
    logger: Any = Field(default_factory=lambda: logging.getLogger(__name__))
    
    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_transformers()
    
    def _initialize_transformers(self) -> None:
        """Initialize coordinate transformers"""
        
        if not PYPROJ_AVAILABLE:
            self.logger.warning("pyproj not available - using simplified transformations")
            return
        
        try:
            # Auto-detect UTM zone if needed
            if self.target_crs == CoordinateSystem.UTM_AUTO.value:
                self.target_crs = self._detect_utm_zone()
            
            # Create transformers
            self._transformer = Transformer.from_crs(
                self.source_crs, self.target_crs, always_xy=True
            )
            self._inverse_transformer = Transformer.from_crs(
                self.target_crs, self.source_crs, always_xy=True
            )
            
            self.logger.info(f"Initialized coordinate transformer: {self.source_crs} -> {self.target_crs}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize coordinate transformers: {e}")
            self._transformer = None
            self._inverse_transformer = None
    
    def _detect_utm_zone(self, center_lon: float = 0.0) -> str:
        """Detect appropriate UTM zone for given longitude"""
        
        # Default to zone 33N if no longitude provided
        if center_lon == 0.0:
            return "EPSG:32633"
        
        # Calculate UTM zone
        zone = int((center_lon + 186) / 6)
        
        # Assume northern hemisphere for simplicity
        epsg_code = 32600 + zone
        
        return f"EPSG:{epsg_code}"
    
    def transform_point(self, x: float, y: float, z: float = 0.0) -> Tuple[float, float, float]:
        """Transform a single point"""
        
        if self._transformer is None:
            # Simple offset transformation for local coordinates
            return (x - 0.0, y - 0.0, z)
        
        try:
            tx, ty = self._transformer.transform(x, y)
            return (tx, ty, z)
        except Exception as e:
            self.logger.error(f"Point transformation failed: {e}")
            return (x, y, z)
    
    def transform_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Transform multiple points"""
        
        if not points:
            return []
        
        if self._transformer is None:
            return [(x - 0.0, y - 0.0) for x, y in points]
        
        try:
            xs, ys = zip(*points)
            txs, tys = self._transformer.transform(xs, ys)
            return list(zip(txs, tys))
        except Exception as e:
            self.logger.error(f"Batch point transformation failed: {e}")
            return points
    
    def inverse_transform_point(self, x: float, y: float, z: float = 0.0) -> Tuple[float, float, float]:
        """Transform point back to source coordinate system"""
        
        if self._inverse_transformer is None:
            return (x + 0.0, y + 0.0, z)
        
        try:
            tx, ty = self._inverse_transformer.transform(x, y)
            return (tx, ty, z)
        except Exception as e:
            self.logger.error(f"Inverse point transformation failed: {e}")
            return (x, y, z)
    
    def get_bounds_in_target_crs(self, bounds: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Transform bounds to target CRS"""
        
        min_x, min_y, max_x, max_y = bounds
        
        # Transform corner points
        corners = [
            (min_x, min_y),
            (min_x, max_y),
            (max_x, min_y),
            (max_x, max_y)
        ]
        
        transformed_corners = self.transform_points(corners)
        
        if transformed_corners:
            xs, ys = zip(*transformed_corners)
            return (min(xs), min(ys), max(xs), max(ys))
        
        return bounds

class RasterProcessor(BaseModel):
    """Process raster datasets for spatial analysis"""
    
    model_config = {"validate_assignment": True}
    
    # Configuration
    coordinate_transformer: CoordinateTransformer
    default_resolution: float = 10.0  # meters per pixel
    
    # Logger
    logger: Any = Field(default_factory=lambda: logging.getLogger(__name__))
    
    def __init__(self, **data):
        super().__init__(**data)
    
    def load_raster_data(self, file_path: str) -> Tuple[np.ndarray, RasterMetadata]:
        """Load raster data from file"""
        
        if not RASTERIO_AVAILABLE:
            raise RuntimeError("rasterio required for raster processing")
        
        try:
            with rasterio.open(file_path) as dataset:
                # Read all bands
                data = dataset.read()
                
                # Create metadata
                transform = GeoTransform(
                    source_crs=str(dataset.crs),
                    target_crs=self.coordinate_transformer.target_crs,
                    bounds=dataset.bounds
                )
                
                metadata = RasterMetadata(
                    width=dataset.width,
                    height=dataset.height,
                    bands=dataset.count,
                    data_type=str(dataset.dtypes[0]),
                    crs=str(dataset.crs),
                    transform=transform,
                    nodata_value=dataset.nodata
                )
                
                # Calculate statistics for each band
                for i in range(dataset.count):
                    band_data = data[i]
                    valid_data = band_data[band_data != dataset.nodata] if dataset.nodata else band_data
                    
                    if valid_data.size > 0:
                        metadata.statistics[f"band_{i+1}"] = {
                            "min": float(np.min(valid_data)),
                            "max": float(np.max(valid_data)),
                            "mean": float(np.mean(valid_data)),
                            "std": float(np.std(valid_data))
                        }
                
                self.logger.info(f"Loaded raster: {metadata.width}x{metadata.height}, {metadata.bands} bands")
                return data, metadata
                
        except Exception as e:
            self.logger.error(f"Failed to load raster data from {file_path}: {e}")
            raise
    
    def extract_land_cover_patches(self, raster_data: np.ndarray, 
                                 metadata: RasterMetadata,
                                 land_cover_mapping: Dict[int, LandCoverType]) -> List[SpatialPatch]:
        """Extract spatial patches from land cover raster"""
        
        patches = []
        patch_id = 0
        
        # Use first band for land cover classification
        land_cover = raster_data[0] if raster_data.ndim == 3 else raster_data
        
        # Get unique land cover values
        unique_values = np.unique(land_cover)
        
        for value in unique_values:
            if value == metadata.nodata_value:
                continue
            
            land_cover_type = land_cover_mapping.get(int(value), LandCoverType.MIXED)
            
            # Create mask for this land cover type
            mask = (land_cover == value).astype(np.uint8)
            
            if not RASTERIO_AVAILABLE:
                # Simplified patch creation without rasterio
                patch_center = SpatialPoint(x=0.0, y=0.0, z=0.0)
                patch_area = float(np.sum(mask)) * (self.default_resolution ** 2) / 10000  # hectares
                
                patch = SpatialPatch(
                    patch_id=patch_id,
                    center=patch_center,
                    area=patch_area,
                    quality=self._land_cover_to_quality(land_cover_type),
                    resource_density=self._land_cover_to_resource_density(land_cover_type)
                )
                patches.append(patch)
                patch_id += 1
                continue
            
            # Extract shapes using rasterio
            try:
                # Create a dummy transform for rasterio
                transform = from_bounds(
                    metadata.transform.bounds[0], metadata.transform.bounds[1],
                    metadata.transform.bounds[2], metadata.transform.bounds[3],
                    metadata.width, metadata.height
                )
                
                shapes_gen = shapes(mask, mask=mask, transform=transform)
                
                for shape, shape_value in shapes_gen:
                    if shape_value == 1:  # Valid patch
                        # Calculate patch properties
                        if shape['type'] == 'Polygon':
                            coordinates = shape['coordinates'][0]  # Exterior ring
                            
                            # Calculate centroid
                            xs = [coord[0] for coord in coordinates]
                            ys = [coord[1] for coord in coordinates]
                            center_x = sum(xs) / len(xs)
                            center_y = sum(ys) / len(ys)
                            
                            # Transform to target CRS
                            tx, ty, tz = self.coordinate_transformer.transform_point(center_x, center_y, 0.0)
                            patch_center = SpatialPoint(x=tx, y=ty, z=tz)
                            
                            # Calculate area (simplified)
                            patch_area = self._calculate_polygon_area(coordinates) / 10000  # hectares
                            
                            # Create vertices
                            vertices = []
                            for coord in coordinates[:-1]:  # Exclude duplicate last point
                                vtx, vty, vtz = self.coordinate_transformer.transform_point(coord[0], coord[1], 0.0)
                                vertices.append(SpatialPoint(x=vtx, y=vty, z=vtz))
                            
                            patch = SpatialPatch(
                                patch_id=patch_id,
                                center=patch_center,
                                vertices=vertices,
                                area=patch_area,
                                perimeter=self._calculate_polygon_perimeter(coordinates),
                                quality=self._land_cover_to_quality(land_cover_type),
                                resource_density=self._land_cover_to_resource_density(land_cover_type)
                            )
                            patches.append(patch)
                            patch_id += 1
                            
            except Exception as e:
                self.logger.warning(f"Failed to extract shapes for land cover {value}: {e}")
                continue
        
        self.logger.info(f"Extracted {len(patches)} patches from land cover raster")
        return patches
    
    def _land_cover_to_quality(self, land_cover_type: LandCoverType) -> float:
        """Convert land cover type to habitat quality"""
        
        quality_mapping = {
            LandCoverType.FOREST: 0.8,
            LandCoverType.GRASSLAND: 0.7,
            LandCoverType.AGRICULTURAL: 0.6,
            LandCoverType.WETLAND: 0.5,
            LandCoverType.MIXED: 0.5,
            LandCoverType.DEVELOPED: 0.2,
            LandCoverType.URBAN: 0.1,
            LandCoverType.WATER: 0.0,
            LandCoverType.BARREN: 0.1
        }
        
        return quality_mapping.get(land_cover_type, 0.3)
    
    def _land_cover_to_resource_density(self, land_cover_type: LandCoverType) -> float:
        """Convert land cover type to resource density"""
        
        density_mapping = {
            LandCoverType.FOREST: 0.6,
            LandCoverType.GRASSLAND: 0.8,
            LandCoverType.AGRICULTURAL: 0.9,
            LandCoverType.WETLAND: 0.4,
            LandCoverType.MIXED: 0.6,
            LandCoverType.DEVELOPED: 0.2,
            LandCoverType.URBAN: 0.1,
            LandCoverType.WATER: 0.0,
            LandCoverType.BARREN: 0.0
        }
        
        return density_mapping.get(land_cover_type, 0.3)
    
    def _calculate_polygon_area(self, coordinates: List[Tuple[float, float]]) -> float:
        """Calculate polygon area using shoelace formula"""
        
        if len(coordinates) < 3:
            return 0.0
        
        area = 0.0
        n = len(coordinates)
        
        for i in range(n):
            j = (i + 1) % n
            area += coordinates[i][0] * coordinates[j][1]
            area -= coordinates[j][0] * coordinates[i][1]
        
        return abs(area) / 2.0
    
    def _calculate_polygon_perimeter(self, coordinates: List[Tuple[float, float]]) -> float:
        """Calculate polygon perimeter"""
        
        if len(coordinates) < 2:
            return 0.0
        
        perimeter = 0.0
        
        for i in range(len(coordinates) - 1):
            dx = coordinates[i+1][0] - coordinates[i][0]
            dy = coordinates[i+1][1] - coordinates[i][1]
            perimeter += (dx*dx + dy*dy)**0.5
        
        return perimeter

class VectorProcessor(BaseModel):
    """Process vector datasets for spatial analysis"""
    
    model_config = {"validate_assignment": True}
    
    # Configuration
    coordinate_transformer: CoordinateTransformer
    
    # Logger
    logger: Any = Field(default_factory=lambda: logging.getLogger(__name__))
    
    def __init__(self, **data):
        super().__init__(**data)
    
    def load_vector_data(self, file_path: str, data_format: DataFormat) -> Tuple[List[Dict[str, Any]], VectorMetadata]:
        """Load vector data from various formats"""
        
        if data_format == DataFormat.GEOJSON:
            return self._load_geojson(file_path)
        elif data_format == DataFormat.SHAPEFILE:
            return self._load_shapefile(file_path)
        elif data_format == DataFormat.KML:
            return self._load_kml(file_path)
        elif data_format == DataFormat.CSV_POINTS:
            return self._load_csv_points(file_path)
        elif data_format == DataFormat.GPKG:
            return self._load_geopackage(file_path)
        else:
            raise ValueError(f"Unsupported vector format: {data_format}")
    
    def _load_geojson(self, file_path: str) -> Tuple[List[Dict[str, Any]], VectorMetadata]:
        """Load GeoJSON file"""
        
        try:
            with open(file_path, 'r') as f:
                geojson_data = json.load(f)
            
            features = geojson_data.get('features', [])
            
            # Extract metadata
            geometry_types = set()
            bounds = [float('inf'), float('inf'), float('-inf'), float('-inf')]
            
            processed_features = []
            
            for feature in features:
                geometry = feature.get('geometry', {})
                properties = feature.get('properties', {})
                
                if geometry:
                    geometry_types.add(geometry.get('type', 'Unknown'))
                    
                    # Update bounds
                    coords = self._extract_coordinates_from_geometry(geometry)
                    if coords:
                        xs, ys = zip(*coords)
                        bounds[0] = min(bounds[0], min(xs))
                        bounds[1] = min(bounds[1], min(ys))
                        bounds[2] = max(bounds[2], max(xs))
                        bounds[3] = max(bounds[3], max(ys))
                    
                    # Transform coordinates if needed
                    transformed_geometry = self._transform_geometry(geometry)
                    
                    processed_features.append({
                        'geometry': transformed_geometry,
                        'properties': properties
                    })
            
            metadata = VectorMetadata(
                geometry_type='/'.join(geometry_types),
                feature_count=len(processed_features),
                crs=self.coordinate_transformer.source_crs,
                bounds=tuple(bounds) if bounds[0] != float('inf') else (0, 0, 0, 0)
            )
            
            self.logger.info(f"Loaded GeoJSON: {metadata.feature_count} features")
            return processed_features, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load GeoJSON from {file_path}: {e}")
            raise
    
    def _load_shapefile(self, file_path: str) -> Tuple[List[Dict[str, Any]], VectorMetadata]:
        """Load Shapefile using geopandas if available"""
        
        if not GEOPANDAS_AVAILABLE:
            raise RuntimeError("geopandas required for shapefile processing")
        
        try:
            gdf = gpd.read_file(file_path)
            
            # Transform to target CRS
            if gdf.crs != self.coordinate_transformer.target_crs:
                gdf = gdf.to_crs(self.coordinate_transformer.target_crs)
            
            features = []
            for idx, row in gdf.iterrows():
                features.append({
                    'geometry': row.geometry.__geo_interface__,
                    'properties': row.drop('geometry').to_dict()
                })
            
            metadata = VectorMetadata(
                geometry_type=str(gdf.geom_type.iloc[0]) if not gdf.empty else 'Unknown',
                feature_count=len(gdf),
                crs=str(gdf.crs),
                bounds=tuple(gdf.total_bounds) if not gdf.empty else (0, 0, 0, 0)
            )
            
            self.logger.info(f"Loaded Shapefile: {metadata.feature_count} features")
            return features, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load Shapefile from {file_path}: {e}")
            raise
    
    def _load_kml(self, file_path: str) -> Tuple[List[Dict[str, Any]], VectorMetadata]:
        """Load KML file (simplified implementation)"""
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Remove namespace for easier parsing
            for elem in root.getiterator():
                if '}' in elem.tag:
                    elem.tag = elem.tag.split('}')[1]
            
            features = []
            bounds = [float('inf'), float('inf'), float('-inf'), float('-inf')]
            
            # Find placemarks
            for placemark in root.findall('.//Placemark'):
                name = placemark.find('name')
                description = placemark.find('description')
                
                properties = {
                    'name': name.text if name is not None else '',
                    'description': description.text if description is not None else ''
                }
                
                # Find geometry
                point = placemark.find('.//Point/coordinates')
                if point is not None:
                    coords_text = point.text.strip()
                    coords = [float(x) for x in coords_text.split(',')]
                    
                    if len(coords) >= 2:
                        lon, lat = coords[0], coords[1]
                        
                        # Transform coordinates
                        tx, ty, tz = self.coordinate_transformer.transform_point(lon, lat, 0.0)
                        
                        geometry = {
                            'type': 'Point',
                            'coordinates': [tx, ty]
                        }
                        
                        features.append({
                            'geometry': geometry,
                            'properties': properties
                        })
                        
                        # Update bounds
                        bounds[0] = min(bounds[0], tx)
                        bounds[1] = min(bounds[1], ty)
                        bounds[2] = max(bounds[2], tx)
                        bounds[3] = max(bounds[3], ty)
            
            metadata = VectorMetadata(
                geometry_type='Point',
                feature_count=len(features),
                crs=self.coordinate_transformer.target_crs,
                bounds=tuple(bounds) if bounds[0] != float('inf') else (0, 0, 0, 0)
            )
            
            self.logger.info(f"Loaded KML: {metadata.feature_count} features")
            return features, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load KML from {file_path}: {e}")
            raise
    
    def _load_csv_points(self, file_path: str) -> Tuple[List[Dict[str, Any]], VectorMetadata]:
        """Load CSV file with point coordinates"""
        
        try:
            import csv
            
            features = []
            bounds = [float('inf'), float('inf'), float('-inf'), float('-inf')]
            
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                # Try to detect delimiter
                dialect = csv.Sniffer().sniff(csvfile.read(1024))
                csvfile.seek(0)
                
                reader = csv.DictReader(csvfile, delimiter=dialect.delimiter)
                
                for row in reader:
                    # Look for coordinate columns
                    lon = lat = None
                    
                    for col_name, value in row.items():
                        col_lower = col_name.lower()
                        if col_lower in ['lon', 'longitude', 'long', 'x']:
                            try:
                                lon = float(value)
                            except ValueError:
                                continue
                        elif col_lower in ['lat', 'latitude', 'y']:
                            try:
                                lat = float(value)
                            except ValueError:
                                continue
                    
                    if lon is not None and lat is not None:
                        # Transform coordinates
                        tx, ty, tz = self.coordinate_transformer.transform_point(lon, lat, 0.0)
                        
                        geometry = {
                            'type': 'Point',
                            'coordinates': [tx, ty]
                        }
                        
                        # Remove coordinate columns from properties
                        properties = {k: v for k, v in row.items() 
                                    if k.lower() not in ['lon', 'longitude', 'long', 'x', 'lat', 'latitude', 'y']}
                        
                        features.append({
                            'geometry': geometry,
                            'properties': properties
                        })
                        
                        # Update bounds
                        bounds[0] = min(bounds[0], tx)
                        bounds[1] = min(bounds[1], ty)
                        bounds[2] = max(bounds[2], tx)
                        bounds[3] = max(bounds[3], ty)
            
            metadata = VectorMetadata(
                geometry_type='Point',
                feature_count=len(features),
                crs=self.coordinate_transformer.target_crs,
                bounds=tuple(bounds) if bounds[0] != float('inf') else (0, 0, 0, 0)
            )
            
            self.logger.info(f"Loaded CSV points: {metadata.feature_count} features")
            return features, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV points from {file_path}: {e}")
            raise
    
    def _load_geopackage(self, file_path: str) -> Tuple[List[Dict[str, Any]], VectorMetadata]:
        """Load GeoPackage file"""
        
        if not GEOPANDAS_AVAILABLE:
            raise RuntimeError("geopandas required for GeoPackage processing")
        
        try:
            # List layers in geopackage
            layers = gpd.list_layers(file_path)
            
            if not layers:
                raise ValueError("No layers found in GeoPackage")
            
            # Use first layer
            layer_name = layers[0]['name']
            gdf = gpd.read_file(file_path, layer=layer_name)
            
            # Transform to target CRS
            if gdf.crs != self.coordinate_transformer.target_crs:
                gdf = gdf.to_crs(self.coordinate_transformer.target_crs)
            
            features = []
            for idx, row in gdf.iterrows():
                features.append({
                    'geometry': row.geometry.__geo_interface__,
                    'properties': row.drop('geometry').to_dict()
                })
            
            metadata = VectorMetadata(
                geometry_type=str(gdf.geom_type.iloc[0]) if not gdf.empty else 'Unknown',
                feature_count=len(gdf),
                crs=str(gdf.crs),
                bounds=tuple(gdf.total_bounds) if not gdf.empty else (0, 0, 0, 0)
            )
            
            self.logger.info(f"Loaded GeoPackage layer '{layer_name}': {metadata.feature_count} features")
            return features, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load GeoPackage from {file_path}: {e}")
            raise
    
    def _extract_coordinates_from_geometry(self, geometry: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Extract coordinate pairs from geometry"""
        
        geom_type = geometry.get('type', '')
        coordinates = geometry.get('coordinates', [])
        
        if geom_type == 'Point':
            return [tuple(coordinates[:2])]
        elif geom_type == 'LineString':
            return [tuple(coord[:2]) for coord in coordinates]
        elif geom_type == 'Polygon':
            # Return exterior ring
            if coordinates and len(coordinates) > 0:
                return [tuple(coord[:2]) for coord in coordinates[0]]
        elif geom_type == 'MultiPolygon':
            # Return coordinates from all polygons
            all_coords = []
            for polygon in coordinates:
                if polygon and len(polygon) > 0:
                    all_coords.extend([tuple(coord[:2]) for coord in polygon[0]])
            return all_coords
        
        return []
    
    def _transform_geometry(self, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Transform geometry coordinates to target CRS"""
        
        geom_type = geometry.get('type', '')
        coordinates = geometry.get('coordinates', [])
        
        if geom_type == 'Point':
            tx, ty, tz = self.coordinate_transformer.transform_point(coordinates[0], coordinates[1], 0.0)
            return {'type': 'Point', 'coordinates': [tx, ty]}
        
        elif geom_type == 'LineString':
            transformed_coords = []
            for coord in coordinates:
                tx, ty, tz = self.coordinate_transformer.transform_point(coord[0], coord[1], 0.0)
                transformed_coords.append([tx, ty])
            return {'type': 'LineString', 'coordinates': transformed_coords}
        
        elif geom_type == 'Polygon':
            transformed_rings = []
            for ring in coordinates:
                transformed_ring = []
                for coord in ring:
                    tx, ty, tz = self.coordinate_transformer.transform_point(coord[0], coord[1], 0.0)
                    transformed_ring.append([tx, ty])
                transformed_rings.append(transformed_ring)
            return {'type': 'Polygon', 'coordinates': transformed_rings}
        
        return geometry
    
    def convert_to_spatial_patches(self, features: List[Dict[str, Any]], 
                                 quality_field: Optional[str] = None,
                                 resource_field: Optional[str] = None) -> List[SpatialPatch]:
        """Convert vector features to spatial patches"""
        
        patches = []
        
        for i, feature in enumerate(features):
            geometry = feature.get('geometry', {})
            properties = feature.get('properties', {})
            
            if geometry.get('type') == 'Point':
                coords = geometry.get('coordinates', [0, 0])
                center = SpatialPoint(x=coords[0], y=coords[1], z=0.0)
                
                # Default area for point features
                area = 1.0
                perimeter = 4.0
                
            elif geometry.get('type') == 'Polygon':
                coordinates = geometry.get('coordinates', [[]])
                if coordinates and len(coordinates) > 0:
                    exterior_ring = coordinates[0]
                    
                    # Calculate centroid
                    xs = [coord[0] for coord in exterior_ring]
                    ys = [coord[1] for coord in exterior_ring]
                    center_x = sum(xs) / len(xs)
                    center_y = sum(ys) / len(ys)
                    center = SpatialPoint(x=center_x, y=center_y, z=0.0)
                    
                    # Calculate area and perimeter
                    area = self._calculate_polygon_area([(coord[0], coord[1]) for coord in exterior_ring]) / 10000  # hectares
                    perimeter = self._calculate_polygon_perimeter([(coord[0], coord[1]) for coord in exterior_ring])
                    
                    # Create vertices
                    vertices = []
                    for coord in exterior_ring[:-1]:  # Exclude duplicate last point
                        vertices.append(SpatialPoint(x=coord[0], y=coord[1], z=0.0))
                else:
                    continue
            else:
                continue  # Skip unsupported geometry types
            
            # Extract quality and resource density from properties
            quality = 0.5  # Default
            resource_density = 0.5  # Default
            
            if quality_field and quality_field in properties:
                try:
                    quality = float(properties[quality_field])
                    quality = max(0.0, min(1.0, quality))  # Clamp to [0,1]
                except (ValueError, TypeError):
                    pass
            
            if resource_field and resource_field in properties:
                try:
                    resource_density = float(properties[resource_field])
                    resource_density = max(0.0, min(1.0, resource_density))  # Clamp to [0,1]
                except (ValueError, TypeError):
                    pass
            
            patch = SpatialPatch(
                patch_id=i,
                center=center,
                vertices=getattr(locals(), 'vertices', []),
                area=area,
                perimeter=perimeter,
                quality=quality,
                resource_density=resource_density
            )
            
            patches.append(patch)
        
        self.logger.info(f"Converted {len(patches)} vector features to spatial patches")
        return patches
    
    def _calculate_polygon_area(self, coordinates: List[Tuple[float, float]]) -> float:
        """Calculate polygon area using shoelace formula"""
        
        if len(coordinates) < 3:
            return 0.0
        
        area = 0.0
        n = len(coordinates)
        
        for i in range(n):
            j = (i + 1) % n
            area += coordinates[i][0] * coordinates[j][1]
            area -= coordinates[j][0] * coordinates[i][1]
        
        return abs(area) / 2.0
    
    def _calculate_polygon_perimeter(self, coordinates: List[Tuple[float, float]]) -> float:
        """Calculate polygon perimeter"""
        
        if len(coordinates) < 2:
            return 0.0
        
        perimeter = 0.0
        
        for i in range(len(coordinates) - 1):
            dx = coordinates[i+1][0] - coordinates[i][0]
            dy = coordinates[i+1][1] - coordinates[i][1]
            perimeter += (dx*dx + dy*dy)**0.5
        
        return perimeter

class GISIntegrationManager(BaseModel):
    """Main GIS integration manager"""
    
    model_config = {"validate_assignment": True}
    
    # Core components
    coordinate_transformer: CoordinateTransformer
    raster_processor: RasterProcessor
    vector_processor: VectorProcessor
    
    # Configuration
    cache_directory: str = Field(default_factory=lambda: tempfile.mkdtemp())
    enable_caching: bool = True
    
    # Logger
    logger: Any = Field(default_factory=lambda: logging.getLogger(__name__))
    
    def __init__(self, **data):
        # Initialize coordinate transformer first
        if 'coordinate_transformer' not in data:
            data['coordinate_transformer'] = CoordinateTransformer()
        
        # Initialize processors with the transformer
        if 'raster_processor' not in data:
            data['raster_processor'] = RasterProcessor(coordinate_transformer=data['coordinate_transformer'])
        
        if 'vector_processor' not in data:
            data['vector_processor'] = VectorProcessor(coordinate_transformer=data['coordinate_transformer'])
        
        super().__init__(**data)
    
    def load_spatial_data_from_config(self, config: Dict[str, Any]) -> List[SpatialPatch]:
        """Load spatial data from configuration"""
        
        patches = []
        
        # Process raster data sources
        for raster_config in config.get('raster_sources', []):
            try:
                raster_patches = self._process_raster_source(raster_config)
                patches.extend(raster_patches)
            except Exception as e:
                self.logger.error(f"Failed to process raster source: {e}")
        
        # Process vector data sources
        for vector_config in config.get('vector_sources', []):
            try:
                vector_patches = self._process_vector_source(vector_config)
                patches.extend(vector_patches)
            except Exception as e:
                self.logger.error(f"Failed to process vector source: {e}")
        
        self.logger.info(f"Loaded {len(patches)} spatial patches from GIS data")
        return patches
    
    def _process_raster_source(self, config: Dict[str, Any]) -> List[SpatialPatch]:
        """Process a raster data source"""
        
        file_path = config.get('file_path')
        if not file_path or not Path(file_path).exists():
            raise ValueError(f"Raster file not found: {file_path}")
        
        # Load raster data
        raster_data, metadata = self.raster_processor.load_raster_data(file_path)
        
        # Get land cover mapping
        land_cover_mapping = {}
        for value, land_cover_name in config.get('land_cover_mapping', {}).items():
            try:
                land_cover_type = LandCoverType(land_cover_name.lower())
                land_cover_mapping[int(value)] = land_cover_type
            except (ValueError, KeyError):
                self.logger.warning(f"Unknown land cover type: {land_cover_name}")
        
        # Extract patches
        patches = self.raster_processor.extract_land_cover_patches(
            raster_data, metadata, land_cover_mapping
        )
        
        return patches
    
    def _process_vector_source(self, config: Dict[str, Any]) -> List[SpatialPatch]:
        """Process a vector data source"""
        
        file_path = config.get('file_path')
        if not file_path or not Path(file_path).exists():
            raise ValueError(f"Vector file not found: {file_path}")
        
        # Determine data format
        file_extension = Path(file_path).suffix.lower()
        format_mapping = {
            '.geojson': DataFormat.GEOJSON,
            '.json': DataFormat.GEOJSON,
            '.shp': DataFormat.SHAPEFILE,
            '.kml': DataFormat.KML,
            '.csv': DataFormat.CSV_POINTS,
            '.gpkg': DataFormat.GPKG
        }
        
        data_format = format_mapping.get(file_extension)
        if not data_format:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Load vector data
        features, metadata = self.vector_processor.load_vector_data(file_path, data_format)
        
        # Convert to spatial patches
        quality_field = config.get('quality_field')
        resource_field = config.get('resource_field')
        
        patches = self.vector_processor.convert_to_spatial_patches(
            features, quality_field, resource_field
        )
        
        return patches
    
    def export_spatial_data(self, spatial_index: SpatialIndex, output_path: str, 
                          output_format: DataFormat = DataFormat.GEOJSON) -> str:
        """Export spatial data to various formats"""
        
        if output_format == DataFormat.GEOJSON:
            return self._export_to_geojson(spatial_index, output_path)
        else:
            raise ValueError(f"Export format not supported: {output_format}")
    
    def _export_to_geojson(self, spatial_index: SpatialIndex, output_path: str) -> str:
        """Export spatial patches to GeoJSON"""
        
        features = []
        
        for patch_id, patch in spatial_index._patches.items():
            # Create geometry
            if patch.vertices:
                # Polygon geometry
                coordinates = [[]]
                for vertex in patch.vertices:
                    # Transform back to source CRS for export
                    lon, lat, z = self.coordinate_transformer.inverse_transform_point(
                        vertex.x, vertex.y, vertex.z
                    )
                    coordinates[0].append([lon, lat])
                
                # Close the polygon
                if coordinates[0] and coordinates[0][0] != coordinates[0][-1]:
                    coordinates[0].append(coordinates[0][0])
                
                geometry = {
                    'type': 'Polygon',
                    'coordinates': coordinates
                }
            else:
                # Point geometry
                lon, lat, z = self.coordinate_transformer.inverse_transform_point(
                    patch.center.x, patch.center.y, patch.center.z
                )
                geometry = {
                    'type': 'Point',
                    'coordinates': [lon, lat]
                }
            
            # Create feature
            feature = {
                'type': 'Feature',
                'geometry': geometry,
                'properties': {
                    'patch_id': patch.patch_id,
                    'area': patch.area,
                    'perimeter': patch.perimeter,
                    'quality': patch.quality,
                    'resource_density': patch.resource_density,
                    'accessibility': patch.accessibility
                }
            }
            
            features.append(feature)
        
        # Create GeoJSON
        geojson = {
            'type': 'FeatureCollection',
            'crs': {
                'type': 'name',
                'properties': {
                    'name': self.coordinate_transformer.source_crs
                }
            },
            'features': features
        }
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        self.logger.info(f"Exported {len(features)} patches to GeoJSON: {output_path}")
        return output_path


def create_gis_integration_system(source_crs: str = CoordinateSystem.WGS84.value,
                                target_crs: str = CoordinateSystem.LOCAL.value) -> GISIntegrationManager:
    """Factory function to create GIS integration system"""
    
    # Create coordinate transformer
    transformer = CoordinateTransformer(source_crs=source_crs, target_crs=target_crs)
    
    # Create GIS integration manager
    gis_manager = GISIntegrationManager(coordinate_transformer=transformer)
    
    return gis_manager