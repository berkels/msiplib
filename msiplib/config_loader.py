# spell-checker:disable
"""
config_loader.py - Configuration Loader with TOML/YAML Support and Section Handling

Provides a Config class that:
- Loads configuration from TOML or YAML files
- Validates parameters against a Cerberus schema
- Supports sectioned configuration organized by purpose
- Provides easy access to parameters from specific sections

SECTION OVERVIEW:
=================
Parameters are organized by their usage:

[input]         - File paths and I/O settings (Pipeline)
[preprocessing] - Image cropping and rotation (Pipeline)
[motif_atoms]   - Parameters for get_motif_atoms_dof() function
[uce_config]    - UCE_config dict for get_motif_atoms_dof() function
[background]    - Background correction (Pipeline)
"""

from cerberus import Validator
from typing import Any, Dict
import yaml

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # fallback for older Python versions


# ==============================================================================
# SCHEMA DEFINITIONS - Organized by usage context
# ==============================================================================

# ------------------------------------------------------------------------------
# [input] - File paths and I/O settings (used by Pipeline)
# ------------------------------------------------------------------------------
INPUT_SCHEMA = {
    "input_file": {
        "required": False,
        "type": "string",
        "nullable": True,
        "default": None,
        "meta": {"description": "Path to input image file"},
    },
    "save_directory": {
        "required": False,
        "type": "string",
        "meta": {"description": "Output directory for results"},
    },
    "save_name": {
        "required": False,
        "type": "string",
        "nullable": True,
        "default": None,
        "meta": {"description": "Custom name for output files"},
    },
}

# ------------------------------------------------------------------------------
# [preprocessing] - Image cropping and rotation (used by Pipeline)
# ------------------------------------------------------------------------------
PREPROCESSING_SCHEMA = {
    "crop_start": {
        "type": "list",
        "items": [{"type": "integer"}, {"type": "integer"}],
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "[x, y] start position for cropping"},
    },
    "crop_size": {
        "type": "list",
        "items": [{"type": "integer"}, {"type": "integer"}],
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "[width, height] of main crop region"},
    },
    "crop_size_motif": {
        "type": "list",
        "items": [{"type": "integer"}, {"type": "integer"}],
        "required": False,
        "meta": {"description": "[width, height] of motif detection crop"},
    },
    "rotation_nr": {
        "type": "integer",
        "required": False,
        "default": 0,
        "meta": {"description": "Number of 90Â° rotations (0-3)"},
    },
    "suffix": {
        "type": "string",
        "required": False,
        "default": "",
        "meta": {"description": "Suffix for crop/test directory name (e.g. 'T1')"},
    },
}

# ------------------------------------------------------------------------------
# [motif_atoms] - Parameters for get_motif_atoms_dof() function
# These parameters are passed directly to the msiplib function
# Defaults match those used in set_motif_local_and_global()
# ------------------------------------------------------------------------------
MOTIF_ATOMS_SCHEMA = {
    # --- Unit cell vectors ---
    "v1": {
        "type": "list",
        "items": [{"type": ["string", "float"]}, {"type": ["string", "float"]}],
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "First lattice vector [x, y]", "get_motif_atoms_dof": "v1"},
    },
    "v2": {
        "type": "list",
        "items": [{"type": ["string", "float"]}, {"type": ["string", "float"]}],
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "Second lattice vector [x, y]", "get_motif_atoms_dof": "v2"},
    },
    "compute_uv": {
        "type": "boolean",
        "required": False,
        "default": True,
        "meta": {"description": "Automatically compute lattice vectors", "get_motif_atoms_dof": "compute_uv"},
    },
    "align_to_y_axis": {
        "type": "boolean",
        "required": False,
        "default": False,
        "meta": {"description": "Align lattice to y-axis", "get_motif_atoms_dof": "align_to_y_axis"},
    },
    "reduce_v1v2": {
        "type": "boolean",
        "required": False,
        "default": False,
        "meta": {"description": "Reduce unit cell vectors to primitive form"},
    },
    # --- Atom detection ---
    "num_motif_atoms": {
        "type": "integer",
        "min": 1,
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "Number of atoms in motif", "get_motif_atoms_dof": "num_motif_atoms"},
    },
    "num_sigma": {
        "type": "float",
        "min": 0,
        "required": False,
        "default": 4,
        "meta": {"description": "Number of sigma for Gaussian fitting", "get_motif_atoms_dof": "num_sigma"},
    },
    "initial_diameter": {
        "type": "integer",
        "min": 1,
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "Initial atom diameter estimate", "get_motif_atoms_dof": "initial_diameter"},
    },
    "erase_inf_radius": {
        "type": "integer",
        "min": 1,
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "Radius for erasing influence", "get_motif_atoms_dof": "erase_inf_radius"},
    },
    "separation": {
        "type": "integer",
        "min": 1,
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "Minimum separation between atoms", "get_motif_atoms_dof": "separation"},
    },
    "min_center_value": {
        "type": "float",
        "min": 0,
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "Minimum intensity at atom center", "get_motif_atoms_dof": "min_center_value"},
    },
    "use_kmedoids": {
        "type": "boolean",
        "required": False,
        "default": False,
        "meta": {"description": "Use k-mediods instead of k-means to inialize the motif atoms when num_motif_atoms is set", "get_motif_atoms_dof": "use_kmedoids"},
    },
    "nm_per_pixel": {
        "type": ["string", "float"],
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "Scale in nm/pixel (read from file if not set)", "get_motif_atoms_dof": "nm_per_pixel"},
    },
    # --- Energy filtering ---
    "energy_exclusion_factor": {
        "type": "float",
        "min": 1,
        "required": False,
        "default": 1.15,
        "meta": {"description": "Factor for energy-based filtering", "get_motif_atoms_dof": "energy_exclusion_factor"},
    },
    "show_bad_energies": {
        "type": "boolean",
        "required": False,
        "default": False,
        "meta": {"description": "Show atoms with bad energies", "get_motif_atoms_dof": "show_bad_energies"},
    },
    "height_std_factor": {
        "type": "float",
        "min": 1,
        "required": False,
        "default": 2.5,
        "meta": {"description": "Factor for height standard deviation filtering", "get_motif_atoms_dof": "height_std_factor"},
    },
    # --- Visualization (motif) ---
    "plot": {
        "type": "boolean",
        "required": False,
        "default": False,
        "meta": {"description": "Generate plots during motif detection", "get_motif_atoms_dof": "plot"},
    },
    "show_motif_legend": {
        "type": "boolean",
        "required": False,
        "default": True,
        "meta": {"description": "Show legend in motif plots", "get_motif_atoms_dof": "show_motif_legend"},
    },
    "fit_to_reconstruction": {
        "type": "boolean",
        "required": False,
        "default": False,
        "meta": {"description": "Fit motif to reconstruction", "get_motif_atoms_dof": "fit_to_reconstruction"},
    },
}

# ------------------------------------------------------------------------------
# [uce_config] - UCE_config dict for get_motif_atoms_dof() function
# These are passed as the UCE_config parameter
# ------------------------------------------------------------------------------
UCE_CONFIG_SCHEMA = {
    "downscale_factor": {
        "type": "integer",
        "min": 1,
        "required": False,
        "meta": {"description": "Factor for downscaling before UCE", "uce_config_key": "downscale_factor"},
    },
    "uc_find_peaks_factor": {
        "type": "float",
        "min": 0,
        "required": False,
        "meta": {"description": "Factor for peak finding", "uce_config_key": "factor"},
    },
    "uc_write_to_file": {
        "type": "boolean",
        "required": False,
        "meta": {"description": "Write UCE results to file", "uce_config_key": "write_to_file"},
    },
    "uc_uniform_prefilter_size": {
        "type": "integer",
        "min": 0,
        "required": False,
        "meta": {"description": "Uniform prefilter size for UCE", "uce_config_key": "uniform_prefilter_size"},
    },
}

# ------------------------------------------------------------------------------
# [background] - Background correction (used by Pipeline)
# ------------------------------------------------------------------------------
BACKGROUND_SCHEMA = {
    "rolling_ball_radius": {
        "type": "integer",
        "min": 1,
        "required": False,
        "meta": {"description": "Radius for rolling ball background subtraction"},
    },
    "rolling_ball_intensity": {
        "type": "integer",
        "min": 1,
        "required": False,
        "meta": {"description": "Intensity for rolling ball algorithm"},
    },
}

# ------------------------------------------------------------------------------
# [visualization] - General visualization settings (used by Pipeline)
# ------------------------------------------------------------------------------
VISUALIZATION_SCHEMA = {
    "legend_fontsize": {
        "type": "integer",
        "min": 1,
        "required": False,
        "meta": {"description": "Font size for legends in output plots"},
    },
}

# ------------------------------------------------------------------------------
# [synthetic] - Synthetic image and deformation parameters
# ------------------------------------------------------------------------------
SYNTHETIC_SCHEMA = {
    "v": {
        "type": "list",
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "Lattice vectors for synthetic crystal image"},
    },
    "im_size": {
        "type": "list",
        "items": [{"type": "integer"}, {"type": "integer"}],
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "[width, height] of synthetic image"},
    },
    "n": {
        "type": "integer",
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "Number of atoms in synthetic motif"},
    },
    "g": {
        "type": "list",
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "Motif atom parameters for synthetic image"},
    },
    "A": {
        "type": "list",
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "2x2 linear deformation matrix"},
    },
    "b": {
        "type": "list",
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "2D translation vector for deformation"},
    },
    "alpha": {
        "type": "float",
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "Alpha parameter for nonlinear displacement"},
    },
    "theta": {
        "type": "float",
        "required": False,
        "nullable": True,
        "default": None,
        "meta": {"description": "Rotation angle (degrees) for linear deformation"},
    },
}

# Mapping of section names to their schemas
SECTION_SCHEMAS = {
    "input": INPUT_SCHEMA,
    "preprocessing": PREPROCESSING_SCHEMA,
    "motif_atoms": MOTIF_ATOMS_SCHEMA,      # -> get_motif_atoms_dof() parameters
    "uce_config": UCE_CONFIG_SCHEMA,         # -> UCE_config dict for get_motif_atoms_dof()
    "background": BACKGROUND_SCHEMA,
    "visualization": VISUALIZATION_SCHEMA,
    "synthetic": SYNTHETIC_SCHEMA,
}

# ==============================================================================
# CONFIG CLASS
# ==============================================================================

class Config:
    """
    Configuration loader and validator for TOML/YAML files.
    
    Supports two modes:
    1. Sectioned TOML: Parameters organized in [section] blocks
    2. Flat YAML: All parameters at root level (backward compatible)
    
    Usage:
        # Load from file
        config = Config.from_file("params.toml")
        
        # Access parameters
        input_file = config.get("input_file")
        crop_size = config.get("crop_size", section="crop")
        
        # Get entire section
        crop_params = config.get_section("crop")
        
        # Get flattened dict (all sections merged)
        all_params = config.to_flat_dict()
    """
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize Config with validated data.
        
        Args:
            data: Configuration dictionary (sectioned or flat)
            is_sectioned: Whether data is organized in sections
        """
        self._data = data
        self._validated = False
        self._errors: Dict[str, Any] = {}
    
    @classmethod
    def from_toml(cls, filepath: str) -> "Config":
        """
        Load configuration from a TOML file (sectioned).
        
        Args:
            filepath: Path to TOML configuration file
            
        Returns:
            Validated Config object
            
        Raises:
            ValueError: If validation fails
        """
        with open(filepath, "rb") as f:
            data = tomllib.load(f)
        
        config = cls(data)
        config.validate()
        return config
    
    @classmethod
    def from_yaml(cls, filepath: str) -> "Config":
        """
        Load configuration from a YAML file (flat).
        
        Automatically converts flat YAML structure to sectioned format
        by mapping parameters to their respective sections.
        
        Args:
            filepath: Path to YAML configuration file
            
        Returns:
            Validated Config object
            
        Raises:
            ValueError: If validation fails
        """
        with open(filepath, "r") as f:
            flat_data = yaml.load(f, Loader=yaml.SafeLoader)
        
        # Convert flat data to sectioned format
        sectioned_data = cls._flat_to_sectioned(flat_data)
        
        config = cls(sectioned_data)
        config.validate()
        return config

    @classmethod
    def from_dict(cls, data: Dict[str, Any], validate: bool = True) -> "Config":
        """
        Create a Config from an in-memory dict.

        Supports both:
        - Sectioned dicts: {"input": {...}, "preprocessing": {...}, ...}
        - Flat dicts (legacy): {"input_file": "...", "crop_start": [...], ...}

        Args:
            data: Configuration as a dict.
            validate: If True, validate immediately and apply schema defaults.

        Returns:
            Config instance.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Config.from_dict expected dict, got {type(data).__name__}")

        # Heuristic: treat as sectioned if all keys are known sections and values are dict-like.
        is_sectioned = all(k in SECTION_SCHEMAS for k in data.keys()) and all(
            (v is None or isinstance(v, dict)) for v in data.values()
        )

        if is_sectioned:
            sectioned_data: Dict[str, Any] = {}
            for section_name in SECTION_SCHEMAS.keys():
                section_value = data.get(section_name, {})
                if section_value is None:
                    section_value = {}
                if not isinstance(section_value, dict):
                    raise TypeError(
                        f"Section '{section_name}' must be a dict, got {type(section_value).__name__}"
                    )
                sectioned_data[section_name] = section_value
        else:
            sectioned_data = cls._flat_to_sectioned(data)

        config = cls(sectioned_data)
        if validate:
            config.validate()
        return config
    
    @staticmethod
    def _flat_to_sectioned(flat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert flat YAML data to sectioned format.
        
        Maps each parameter to its corresponding section based on
        which schema contains the parameter key.
        
        Args:
            flat_data: Flat dictionary with all parameters at root level
            
        Returns:
            Dictionary organized by sections
        """
        sectioned = {section: {} for section in SECTION_SCHEMAS.keys()}
        unmatched = {}
        
        for key, value in flat_data.items():
            matched = False
            for section_name, section_schema in SECTION_SCHEMAS.items():
                if key in section_schema:
                    sectioned[section_name][key] = value
                    matched = True
                    break
            
            if not matched:
                # Keep unmatched keys for potential error reporting
                unmatched[key] = value
        
        if unmatched:
            raise TypeError(f"Warning: Unknown parameters: {list(unmatched.keys())}")

        
        return sectioned
    
    
    def validate(self) -> None:
        """
        Validate configuration against schema.
        
        For sectioned configs, validates each section separately.
        For flat configs, validates against combined schema.
            
        Raises:
            ValueError: If validation fails
        """
        self._errors = {}
        
        
        # Validate each section
        validated_data = {}
        for section_name, section_schema in SECTION_SCHEMAS.items():
            section_data = self._data.get(section_name, {})
            validator = Validator(section_schema)
            
            if not validator.validate(section_data):
                self._errors[section_name] = validator.errors
            else:
                validated_data[section_name] = validator.document
        
        if self._errors:
            raise ValueError(f"Configuration validation failed:\n{self._format_errors()}")
        
        self._data = validated_data
        
        self._validated = True
       
    
    def _format_errors(self) -> str:
        """Format validation errors for display."""
        lines = []
        for key, value in self._errors.items():
            lines.append(f"  [{key}]: {value}")
        return "\n".join(lines)
    
    def get(self, key: str, section: str, default: Any = None) -> Any:
        """
        Get a parameter value.
        
        Args:
            key: Parameter name
            section: Section name
            default: Default value if key not found
            
        Returns:
            Parameter value or default
        """
        return self._data.get(section, {}).get(key, default)
        
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get all parameters from a section.
        
        Args:
            section: Section name
            
        Returns:
            Dictionary of parameters in that section
        """
        return self._data.get(section, {}).copy()
    
    
    # =========================================================================
    # CONVENIENCE METHODS for common parameter groups
    # =========================================================================
    
    def get_input_params(self) -> Dict[str, Any]:
        """Get input/output related parameters."""
        return self.get_section("input")
    
    def get_preprocessing_params(self) -> Dict[str, Any]:
        """Get preprocessing (crop/rotation) parameters."""
        return self.get_section("preprocessing")
    
    def get_motif_atoms_params(self) -> Dict[str, Any]:
        """
        Get parameters for get_motif_atoms_dof() function.
        
        Returns dict with all motif detection parameters using their
        expected names for the function call.
        """
        return self.get_section("motif_atoms")
    
    def get_uce_config(self) -> Dict[str, Any]:
        """
        Get UCE_config dict for get_motif_atoms_dof() function.
        
        Returns dict with keys renamed to match UCE_config expected format:
        - uc_find_peaks_factor -> factor
        - uc_write_to_file -> write_to_file
        - uc_uniform_prefilter_size -> uniform_prefilter_size
        """
        params = self.get_section("uce_config")
        uce_config = {}
        if "downscale_factor" in params:
            uce_config["downscale_factor"] = params["downscale_factor"]
        if "uc_find_peaks_factor" in params:
            uce_config["factor"] = params["uc_find_peaks_factor"]
        if "uc_write_to_file" in params:
            uce_config["write_to_file"] = params["uc_write_to_file"]
        if "uc_uniform_prefilter_size" in params:
            uce_config["uniform_prefilter_size"] = params["uc_uniform_prefilter_size"]
        return uce_config
    
    def get_visualization_params(self) -> Dict[str, Any]:
        """Get visualization related parameters."""
        return self.get_section("visualization")
    
    def get_background_params(self) -> Dict[str, Any]:
        """Get background correction parameters."""
        return self.get_section("background")

    def get_synthetic_params(self) -> Dict[str, Any]:
        """Get synthetic image and deformation parameters."""
        return self.get_section("synthetic")
    
    def save_as_toml(self, filepath: str, include_defaults: bool = False) -> None:
        """
        Save current configuration as TOML file.
        
        Useful for converting YAML configs to sectioned TOML format.
        
        Args:
            filepath: Path where to save the TOML file
            include_defaults: If True, include all default values in output
        """
        lines = []
        lines.append("# Configuration file for Motif-Shift analysis")
        lines.append("# Auto-generated from Config object")
        lines.append("")
        
        for section_name in SECTION_SCHEMAS.keys():
            section_data = self._data.get(section_name, {})
            
            # Skip empty sections unless we want defaults
            if not section_data and not include_defaults:
                continue
            
            # Get schema for default values
            schema = SECTION_SCHEMAS.get(section_name, {})
            
            # Collect values to write
            values_to_write = {}
            for key, value in section_data.items():
                # Skip None values unless include_defaults
                if value is None and not include_defaults:
                    continue
                values_to_write[key] = value
            
            # Skip section if nothing to write
            if not values_to_write:
                continue
            
            lines.append(f"[{section_name}]")
            
            for key, value in values_to_write.items():
                toml_value = self._to_toml_value(value)
                
                # Add description as comment if available
                if key in schema and "meta" in schema[key]:
                    desc = schema[key]["meta"].get("description", "")
                    if desc:
                        lines.append(f"# {desc}")
                
                lines.append(f"{key} = {toml_value}")
            
            lines.append("")
        
        with open(filepath, "w") as f:
            f.write("\n".join(lines))
        
        print(f"Configuration saved to: {filepath}")
    
    def to_dict(self, include_defaults: bool = False, include_none: bool = False) -> Dict[str, Any]:
        """
        Return the configuration as a plain (JSON/YAML-serializable) dict.

        By default, this returns a *minimal* representation:
        - Drops keys with value None
        - Drops keys that are equal to the schema default
        - Drops empty sections

        Notes:
            This is an approximation of "only parameters that were set".
            If a user explicitly sets a value equal to the default, it cannot
            be distinguished from an implicit default after validation.
        """

        minimal: Dict[str, Any] = {}
        for section_name, section_values in self._data.items():
            schema = SECTION_SCHEMAS.get(section_name, {})
            filtered_section: Dict[str, Any] = {}

            for key, value in section_values.items():
                if value is None and not include_none:
                    continue

                if not include_defaults and key in schema and "default" in schema[key]:
                    if value == schema[key]["default"]:
                        continue

                filtered_section[key] = value

            if filtered_section:
                minimal[section_name] = filtered_section

        return minimal
    
    @staticmethod
    def _to_toml_value(value: Any) -> str:
        """Convert Python value to TOML string representation."""
        if value is None:
            return "# None (not set)"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, str):
            # Escape quotes and wrap in quotes
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            items = [Config._to_toml_value(item) for item in value]
            return f"[{', '.join(items)}]"
        elif isinstance(value, dict):
            # Inline table for simple dicts
            items = [f"{k} = {Config._to_toml_value(v)}" for k, v in value.items()]
            return "{ " + ", ".join(items) + " }"
        else:
            return str(value)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def create_example_toml(filepath: str):
    """
    Create an example TOML configuration file.
    
    Args:
        filepath: Path where to save the example file
    """
    example = '''# Configuration file for Motif-Shift analysis
# ============================================
# 
# Sections are organized by usage:
# - [input]         : File paths and I/O (Pipeline)
# - [preprocessing] : Image cropping/rotation (Pipeline)
# - [motif_atoms]   : Parameters for get_motif_atoms_dof()
# - [uce_config]    : UCE_config dict for get_motif_atoms_dof()
# - [background]    : Background correction (Pipeline)
# - [visualization] : General plot settings (Pipeline)

# ==============================================================================
# [input] - File paths (used by Pipeline)
# ==============================================================================
[input]
input_file = "${SCIEBO_DIR}/emic/data/example.nc"
save_directory = "${OUTPUT_DIR}/Border_detection/"
# save_name = "custom_name"

# ==============================================================================
# [preprocessing] - Image cropping and rotation (used by Pipeline)
# ==============================================================================
[preprocessing]
crop_start = [100, 100]
crop_size = [512, 512]
crop_size_motif = [256, 256]
rotation_nr = 0  # default: 0

# ==============================================================================
# [motif_atoms] - Parameters for get_motif_atoms_dof() function
# These are passed directly to msiplib.motif_atoms_dof.get_motif_atoms_dof()
# ==============================================================================
[motif_atoms]
# --- Unit cell vectors ---
compute_uv = true           # default: True - auto-compute lattice vectors
# v1 = [10.0, 0.0]          # default: None - provide if known
# v2 = [0.0, 10.0]          # default: None - provide if known
align_to_y_axis = false     # default: False

# --- Atom detection ---
# n = 2                     # default: None - number of atoms in motif
num_sigma = 4               # default: 4 - sigma for Gaussian fitting
# initial_diameter = 10     # default: None
# erase_inf_radius = 5      # default: None
# separation = 5            # default: None - min separation between atoms
# min_center_value = 0.1    # default: None - min intensity at center
# nm_per_pixel = 0.01       # default: None - read from file if not set

# --- Energy filtering ---
energy_exclusion_factor = 1.15  # default: 1.15
show_bad_energies = false       # default: False
height_std_factor = 2.5         # default: 2.5

# --- Visualization (motif detection) ---
plot = false                    # default: False
show_motif_legend = true        # default: True

# ==============================================================================
# [uce_config] - UCE_config dict for get_motif_atoms_dof()
# ==============================================================================
[uce_config]
# downscale_factor = 2          # for faster UCE computation
# uc_find_peaks_factor = 0.5    # peak finding sensitivity
# uc_write_to_file = false      # save UCE results
# uc_uniform_prefilter_size = 3 # prefilter size

# ==============================================================================
# [background] - Background correction (used by Pipeline)
# ==============================================================================
[background]
# rolling_ball_radius = 50
# rolling_ball_intensity = 255

# ==============================================================================
# [visualization] - General plot settings (used by Pipeline)
# ==============================================================================
[visualization]
# legend_fontsize = 12
'''
    with open(filepath, "w") as f:
        f.write(example)
    print(f"Example TOML config written to: {filepath}")


# ==============================================================================
# MAIN - for testing
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Load and display config from file
        config = Config.from_toml(sys.argv[1])
        print(f"Loaded: {config}")
    else:
        # Create example file
        create_example_toml("example_config.toml")
