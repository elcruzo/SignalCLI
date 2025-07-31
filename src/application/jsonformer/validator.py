"""JSON schema validator and formatter."""

import json
from typing import Dict, Any, Optional, List, Union
from jsonschema import validate, ValidationError as JsonSchemaError, Draft7Validator

from src.core.exceptions import JSONFormerError, SchemaValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

class JSONValidator:
    """Validates and formats JSON according to schema."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_string_length = config.get('max_string_length', 500)
        self.max_array_length = config.get('max_array_length', 20)
        self.max_object_properties = config.get('max_object_properties', 50)
        
    def validate_and_format(
        self,
        data: Union[str, Dict[str, Any]],
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and format data according to schema.
        
        Args:
            data: JSON string or dict to validate
            schema: JSON schema
            
        Returns:
            Validated and formatted data
            
        Raises:
            SchemaValidationError: If validation fails
        """
        try:
            # Parse if string
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError as e:
                    raise SchemaValidationError(f"Invalid JSON: {e}")
                    
            # Validate schema itself
            self._validate_schema(schema)
            
            # Apply constraints
            data = self._apply_constraints(data, schema)
            
            # Validate data against schema
            validate(instance=data, schema=schema)
            
            # Format output
            formatted = self._format_output(data, schema)
            
            return formatted
            
        except JsonSchemaError as e:
            logger.error(f"Schema validation failed: {e}")
            raise SchemaValidationError(f"Validation failed: {e.message}")
        except Exception as e:
            logger.error(f"Validation error: {e}")
            raise JSONFormerError(f"Validation error: {e}")
    
    def _validate_schema(self, schema: Dict[str, Any]) -> None:
        """Validate the schema itself."""
        try:
            Draft7Validator.check_schema(schema)
        except JsonSchemaError as e:
            raise SchemaValidationError(f"Invalid schema: {e}")
            
    def _apply_constraints(
        self,
        data: Any,
        schema: Dict[str, Any]
    ) -> Any:
        """Apply size constraints to data."""
        schema_type = schema.get('type')
        
        if schema_type == 'string':
            if isinstance(data, str) and len(data) > self.max_string_length:
                data = data[:self.max_string_length]
                logger.warning(f"Truncated string to {self.max_string_length} chars")
                
        elif schema_type == 'array':
            if isinstance(data, list) and len(data) > self.max_array_length:
                data = data[:self.max_array_length]
                logger.warning(f"Truncated array to {self.max_array_length} items")
                
        elif schema_type == 'object':
            if isinstance(data, dict):
                # Apply property constraints
                properties = schema.get('properties', {})
                required = schema.get('required', [])
                
                # Remove extra properties if additionalProperties is false
                if not schema.get('additionalProperties', True):
                    data = {k: v for k, v in data.items() if k in properties}
                    
                # Limit number of properties
                if len(data) > self.max_object_properties:
                    # Keep required properties first
                    kept_props = {}
                    for prop in required:
                        if prop in data:
                            kept_props[prop] = data[prop]
                            
                    # Add other properties up to limit
                    for k, v in data.items():
                        if len(kept_props) >= self.max_object_properties:
                            break
                        if k not in kept_props:
                            kept_props[k] = v
                            
                    data = kept_props
                    
                # Recursively apply constraints to properties
                for prop, prop_schema in properties.items():
                    if prop in data:
                        data[prop] = self._apply_constraints(data[prop], prop_schema)
                        
        return data
    
    def _format_output(
        self,
        data: Any,
        schema: Dict[str, Any]
    ) -> Any:
        """Format output according to schema preferences."""
        schema_type = schema.get('type')
        
        if schema_type == 'object' and isinstance(data, dict):
            # Ensure required properties exist
            required = schema.get('required', [])
            properties = schema.get('properties', {})
            
            for prop in required:
                if prop not in data:
                    # Add default value if specified
                    prop_schema = properties.get(prop, {})
                    if 'default' in prop_schema:
                        data[prop] = prop_schema['default']
                    else:
                        # Add null for missing required properties
                        data[prop] = None
                        
            # Format nested properties
            for prop, prop_schema in properties.items():
                if prop in data:
                    data[prop] = self._format_output(data[prop], prop_schema)
                    
        elif schema_type == 'array' and isinstance(data, list):
            # Format array items
            items_schema = schema.get('items', {})
            data = [self._format_output(item, items_schema) for item in data]
            
        elif schema_type == 'number' and isinstance(data, (int, float)):
            # Apply number formatting
            if 'multipleOf' in schema:
                multiple = schema['multipleOf']
                data = round(data / multiple) * multiple
                
        elif schema_type == 'string' and isinstance(data, str):
            # Apply string formatting
            if 'pattern' in schema:
                # Could validate pattern here
                pass
                
            # Trim whitespace
            data = data.strip()
            
        return data
    
    def repair_json(self, json_str: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to repair malformed JSON.
        
        Args:
            json_str: Potentially malformed JSON string
            
        Returns:
            Repaired JSON object or None if unrepairable
        """
        try:
            # First try normal parsing
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
        # Try common repairs
        repairs = [
            # Missing quotes around keys
            lambda s: self._add_quotes_to_keys(s),
            # Trailing commas
            lambda s: s.replace(',]', ']').replace(',}', '}'),
            # Single quotes to double quotes
            lambda s: s.replace("'", '"'),
            # Missing closing braces
            lambda s: self._balance_braces(s),
        ]
        
        for repair in repairs:
            try:
                repaired = repair(json_str)
                return json.loads(repaired)
            except:
                continue
                
        logger.warning("Failed to repair JSON")
        return None
        
    def _add_quotes_to_keys(self, json_str: str) -> str:
        """Add quotes to unquoted keys."""
        import re
        # Simple regex to add quotes around unquoted keys
        pattern = r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:'
        return re.sub(pattern, r'\1"\2":', json_str)
        
    def _balance_braces(self, json_str: str) -> str:
        """Balance opening and closing braces."""
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # Add missing closing braces/brackets
        json_str += '}' * (open_braces - close_braces)
        json_str += ']' * (open_brackets - close_brackets)
        
        return json_str