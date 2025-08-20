# Modular Geotechnical Mapping System

## Overview
A comprehensive modular system for processing and visualizing geotechnical test data with both static PNG and interactive HTML map outputs.

## Module Structure

### 1. `coordinate_transformer.py` - Coordinate Processing
**Purpose**: Handles coordinate cleaning and transformation from British National Grid to WGS84.

**Key Functions**:
- `load_and_clean_data()` - Loads CSV and removes invalid coordinates
- `transform_coordinates()` - Vectorized BNG to WGS84 conversion using pyproj
- `get_coordinate_bounds()` - Calculates site boundaries and dimensions
- `process_csv()` - Complete pipeline for coordinate processing

**Output**: Cleaned CSV with both BNG and WGS84 coordinates

### 2. `quadrant_analyzer.py` - Spatial Analysis
**Purpose**: Divides site into 250m×250m quadrants and creates numpy arrays for analysis.

**Key Functions**:
- `assign_quadrants()` - Maps each test location to its quadrant
- `create_occupancy_matrix()` - Binary numpy array (1=data, 0=empty)
- `get_quadrant_summary()` - Statistical summary per quadrant
- `process_quadrants()` - Complete quadrant analysis pipeline

**Output**: Quadrant assignments, occupancy matrix, and summary statistics

### 3. `map_imagery_producer.py` - Map Generation
**Purpose**: Creates both static PNG and interactive HTML maps with satellite imagery.

**Key Functions**:
- `produce_map()` - Static PNG map with matplotlib and contextily
- `create_interactive_html_map()` - Interactive HTML map with folium
- `produce_maps()` - Generates both PNG and HTML outputs
- Helper functions for quadrant grids, test points, and legends

**Output**: PNG and HTML maps with satellite imagery and data overlays

### 4. `main_workflow.py` - Orchestration
**Purpose**: Coordinates all modules in sequence with comprehensive logging.

## Map Outputs

### Static PNG Map (`sesro_satellite_map.png`)
- High-resolution satellite imagery background
- Color-coded quadrant grid (green=data, orange=empty)
- Test points colored by type (CPT, Hand Vane, Triaxial)
- Professional legends and annotations
- Publication-ready quality

### Interactive HTML Map (`sesro_interactive_map.html`)
- **Toggleable satellite imagery** - Switch between OpenStreetMap and satellite views
- **Layer controls** - Toggle test types and quadrant overlays on/off
- **Interactive popups** - Click on points for detailed test information
- **Zoom and pan** - Full map navigation capabilities
- **Responsive design** - Works on desktop and mobile devices

## Key Features

### Data Processing
- Processes 10,401 valid test locations from 15,650 original records
- Covers 49.1 km² site area (7,515m × 6,537m)
- Systematic 250m×250m grid analysis (837 total quadrants)
- 11.9% data occupancy rate (100 occupied quadrants)

### Test Type Classification
- **CPTDerived**: 9,971 tests (blue markers)
- **HandVane**: 279 tests (orange markers)  
- **TriaxialTotal**: 151 tests (green markers)

### Performance Optimization
- Vectorized coordinate transformations (0.008s for 10,401 points)
- Chunked data processing for memory efficiency
- Systematic numpy-based grid analysis

## Usage

### Run Complete Workflow
```bash
python main_workflow.py
```

### Run Individual Modules
```python
# Coordinate transformation only
from coordinate_transformer import CoordinateTransformer
transformer = CoordinateTransformer()
df, bounds = transformer.process_csv("../CompiledCu.csv")

# Quadrant analysis only
from quadrant_analyzer import QuadrantAnalyzer
analyzer = QuadrantAnalyzer(quadrant_size=250)
df_with_quadrants, matrix, summary = analyzer.process_quadrants(df, bounds)

# Map generation only
from map_imagery_producer import MapImageryProducer
producer = MapImageryProducer()
producer.produce_maps(df, matrix, bounds, "map.png", "map.html")
```

## Output Files

1. **`coordinates_processed.csv`** - Cleaned coordinates with BNG/WGS84
2. **`quadrant_summary.csv`** - Quadrant analysis results
3. **`sesro_satellite_map.png`** - Static high-resolution map
4. **`sesro_interactive_map.html`** - Interactive web map

## Interactive Map Controls

### Base Layers (Radio Selection)
- **OpenStreetMap** - Street map with labels
- **Satellite Imagery** - High-resolution aerial photography

### Data Overlays (Checkbox Toggle)
- **CPTDerived Tests** - Blue circular markers
- **HandVane Tests** - Orange circular markers
- **TriaxialTotal Tests** - Green circular markers
- **Quadrant Grid** - Green (data) / Orange (empty) overlay

### Interactive Features
- **Click popups** - Test location details and coordinates
- **Zoom controls** - Mouse wheel and +/- buttons
- **Pan navigation** - Click and drag map movement
- **Layer control** - Top-right panel for toggling layers
- **Legend** - Bottom-left information panel
- **Statistics** - Top-left summary panel

## Dependencies

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing and arrays
- **pyproj** - Coordinate system transformations
- **matplotlib** - Static map plotting
- **contextily** - Satellite imagery integration
- **folium** - Interactive web maps
- **logging** - Comprehensive workflow tracking

## Technical Notes

- All coordinates processed in British National Grid (EPSG:27700)
- WGS84 (EPSG:4326) used for web map display
- Quadrant system uses systematic grid matrix approach
- HTML maps use Leaflet.js via folium for interactivity
- PNG maps optimized for publication at 300 DPI
