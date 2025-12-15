"""
Tests for data interface classes.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from vdm.data_interface import (
    SimulationLoader,
    HaloCatalogLoader,
    GadgetHDF5Loader,
    SubFindCatalog,
    RockstarCatalog,
    CSVCatalog,
    get_simulation_loader,
    get_halo_catalog,
)


class TestAbstractInterfaces:
    """Test abstract base classes."""
    
    def test_simulation_loader_is_abstract(self):
        """SimulationLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SimulationLoader('/fake/path')
    
    def test_halo_catalog_loader_is_abstract(self):
        """HaloCatalogLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            HaloCatalogLoader('/fake/path')


class TestCSVCatalog:
    """Test CSV halo catalog loader."""
    
    @pytest.fixture
    def csv_catalog(self, tmp_path):
        """Create a temporary CSV catalog."""
        csv_file = tmp_path / "halos.csv"
        
        # Create test data
        csv_content = """x,y,z,mass,radius
10.0,20.0,30.0,1e14,0.5
15.0,25.0,35.0,5e13,0.3
5.0,10.0,15.0,1e12,0.1
"""
        csv_file.write_text(csv_content)
        return csv_file
    
    def test_load_csv_catalog(self, csv_catalog):
        """Test loading CSV catalog."""
        catalog = CSVCatalog(str(csv_catalog))
        positions, masses, radii = catalog.load_halos(mass_threshold=1e13)
        
        assert len(positions) == 2  # Only 2 halos above threshold
        assert positions.shape[1] == 3
        assert len(masses) == 2
        assert len(radii) == 2
    
    def test_csv_mass_threshold(self, csv_catalog):
        """Test mass threshold filtering."""
        catalog = CSVCatalog(str(csv_catalog))
        
        # High threshold
        pos, mass, rad = catalog.load_halos(mass_threshold=5e13)
        assert len(mass) == 2
        
        # Very high threshold
        pos, mass, rad = catalog.load_halos(mass_threshold=2e14)
        assert len(mass) == 0
        
        # Low threshold
        pos, mass, rad = catalog.load_halos(mass_threshold=1e11)
        assert len(mass) == 3
    
    def test_csv_unit_conversion(self, tmp_path):
        """Test unit conversion in CSV loader."""
        csv_file = tmp_path / "halos_kpc.csv"
        
        # Data in kpc
        csv_content = """x,y,z,mass,radius
10000.0,20000.0,30000.0,1e14,500.0
"""
        csv_file.write_text(csv_content)
        
        # Load with unit conversion
        catalog = CSVCatalog(
            str(csv_file),
            units={'position': 0.001, 'radius': 0.001}  # kpc -> Mpc
        )
        positions, masses, radii = catalog.load_halos(mass_threshold=0)
        
        assert positions[0, 0] == pytest.approx(10.0, rel=1e-5)
        assert radii[0] == pytest.approx(0.5, rel=1e-5)


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_get_halo_catalog_csv(self, tmp_path):
        """Test factory function for CSV."""
        csv_file = tmp_path / "halos.csv"
        csv_file.write_text("x,y,z,mass\n1,2,3,1e14\n")
        
        catalog = get_halo_catalog(str(csv_file), format='csv')
        assert isinstance(catalog, CSVCatalog)
    
    def test_get_halo_catalog_auto_csv(self, tmp_path):
        """Test auto-detection of CSV format."""
        csv_file = tmp_path / "halos.csv"
        csv_file.write_text("x,y,z,mass\n1,2,3,1e14\n")
        
        catalog = get_halo_catalog(str(csv_file), format='auto')
        assert isinstance(catalog, CSVCatalog)
    
    def test_get_simulation_loader_unknown_format(self):
        """Test error for unknown format."""
        with pytest.raises(ValueError, match="Unknown format"):
            get_simulation_loader('/fake/path', format='unknown')
    
    def test_get_halo_catalog_unknown_format(self):
        """Test error for unknown format."""
        with pytest.raises(ValueError, match="Unknown format"):
            get_halo_catalog('/fake/path', format='unknown')


class TestDataInterfaceIntegration:
    """Integration tests for data interface."""
    
    def test_csv_to_numpy_pipeline(self, tmp_path):
        """Test full pipeline from CSV to numpy arrays."""
        # Create test catalog
        csv_file = tmp_path / "halos.csv"
        n_halos = 100
        
        np.random.seed(42)
        x = np.random.uniform(0, 50, n_halos)
        y = np.random.uniform(0, 50, n_halos)
        z = np.random.uniform(0, 50, n_halos)
        mass = 10**np.random.uniform(12, 15, n_halos)
        radius = 0.78 * (mass / 1e14)**(1/3)
        
        # Write CSV
        with open(csv_file, 'w') as f:
            f.write("x,y,z,mass,radius\n")
            for i in range(n_halos):
                f.write(f"{x[i]},{y[i]},{z[i]},{mass[i]},{radius[i]}\n")
        
        # Load and verify
        catalog = get_halo_catalog(str(csv_file))
        positions, masses, radii = catalog.load_halos(mass_threshold=1e13)
        
        # Check we got reasonable data
        assert len(masses) > 0
        assert len(masses) <= n_halos
        assert positions.min() >= 0
        assert positions.max() <= 50
        assert masses.min() >= 1e13


class TestCustomImplementation:
    """Test custom implementation of interfaces."""
    
    def test_custom_simulation_loader(self):
        """Test that custom SimulationLoader can be implemented."""
        
        class MyLoader(SimulationLoader):
            def load_particles(self, particle_type):
                return np.zeros((10, 3)), np.ones(10)
            
            def get_box_size(self):
                return 50.0
        
        loader = MyLoader('/fake/path')
        pos, mass = loader.load_particles('dm')
        
        assert pos.shape == (10, 3)
        assert mass.shape == (10,)
        assert loader.get_box_size() == 50.0
    
    def test_custom_halo_catalog(self):
        """Test that custom HaloCatalogLoader can be implemented."""
        
        class MyCatalog(HaloCatalogLoader):
            def load_halos(self, mass_threshold=1e13):
                return (
                    np.array([[25, 25, 25]]),
                    np.array([1e14]),
                    np.array([0.5])
                )
        
        catalog = MyCatalog('/fake/path')
        pos, mass, rad = catalog.load_halos()
        
        assert pos.shape == (1, 3)
        assert mass[0] == 1e14
        assert rad[0] == 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
