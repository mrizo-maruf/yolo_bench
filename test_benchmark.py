#!/usr/bin/env python3
"""
Unit tests for YOLO benchmarking tool.
"""

import unittest
import tempfile
import yaml
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestConfigValidation(unittest.TestCase):
    """Test configuration file validation."""
    
    def test_config_yaml_loads(self):
        """Test that config.yaml can be loaded."""
        config_path = Path(__file__).parent / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.assertIsNotNone(config)
        self.assertIn('dataset', config)
        self.assertIn('models', config)
        self.assertIn('benchmark', config)
        self.assertIn('output', config)
    
    def test_data_yaml_example_loads(self):
        """Test that data.yaml.example can be loaded."""
        example_path = Path(__file__).parent / 'data.yaml.example'
        with open(example_path, 'r') as f:
            data = yaml.safe_load(f)
        
        self.assertIsNotNone(data)
        self.assertIn('path', data)
        self.assertIn('train', data)
        self.assertIn('val', data)
        self.assertIn('names', data)
        self.assertIn('nc', data)


class TestBenchmarkImport(unittest.TestCase):
    """Test that benchmark module can be imported."""
    
    def test_import_benchmark_without_dependencies(self):
        """Test that we can at least parse the benchmark module."""
        import ast
        benchmark_path = Path(__file__).parent / 'benchmark.py'
        with open(benchmark_path, 'r') as f:
            code = f.read()
        
        # This will raise SyntaxError if there are syntax issues
        ast.parse(code)
        self.assertTrue(True)  # If we get here, parsing succeeded


class TestValidateDatasetImport(unittest.TestCase):
    """Test validate_dataset module."""
    
    def test_import_validate_dataset(self):
        """Test that we can at least parse the validate_dataset module."""
        import ast
        validate_path = Path(__file__).parent / 'validate_dataset.py'
        with open(validate_path, 'r') as f:
            code = f.read()
        
        # This will raise SyntaxError if there are syntax issues
        ast.parse(code)
        self.assertTrue(True)


class TestQuickStartScript(unittest.TestCase):
    """Test quick_start module."""
    
    def test_import_quick_start(self):
        """Test that we can at least parse the quick_start module."""
        import ast
        quick_start_path = Path(__file__).parent / 'quick_start.py'
        with open(quick_start_path, 'r') as f:
            code = f.read()
        
        # This will raise SyntaxError if there are syntax issues
        ast.parse(code)
        self.assertTrue(True)


class TestYAMLStructure(unittest.TestCase):
    """Test YAML file structures."""
    
    def test_config_has_required_sections(self):
        """Test that config.yaml has all required sections."""
        config_path = Path(__file__).parent / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check dataset section
        self.assertIn('data_yaml', config['dataset'])
        
        # Check models section
        self.assertIn('yolov8', config['models'])
        self.assertIn('yolov11', config['models'])
        
        # Check benchmark section
        required_bench_keys = ['imgsz', 'batch', 'conf', 'iou', 'device']
        for key in required_bench_keys:
            self.assertIn(key, config['benchmark'])
        
        # Check output section
        required_output_keys = ['results_dir', 'generate_plots', 'save_csv']
        for key in required_output_keys:
            self.assertIn(key, config['output'])


if __name__ == '__main__':
    unittest.main()
