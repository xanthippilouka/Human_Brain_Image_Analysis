#Import required libraries
import cv2
import pandas as pd
import numpy as np
from scipy import stats, ndimage
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure, color
from skimage.color import separate_stains
import seaborn as sns


class QuPathStainVectors:
    
    #Import and use the stain vectors estimated by QuPath
    
    def __init__(self):
        self.hematoxylin_vector = None
        self.dab_vector = None
        self.stain_matrix = None
        self.all_vectors = []  # Store vectors from multiple images
    
    def add_qupath_vectors(self, h_vector, dab_vector, image_name=""):
        """
        Add stain vectors from QuPath for one image (QuPath vectors are already normalized OD vectors)
        
        Parameters:
        h_vector: list or array [R, G, B] for hematoxylin
        dab_vector: list or array [R, G, B] for DAB
        image_name: optional name to track which image these came from
        """
        h_vec = np.array(h_vector, dtype=np.float64)
        dab_vec = np.array(dab_vector, dtype=np.float64)
        
        self.all_vectors.append({
            'image_name': image_name,
            'hematoxylin': h_vec,
            'dab': dab_vec
        })
        
        print(f" Added vectors from: {image_name if image_name else 'Image'}")
        print(f"  H:   [{h_vec[0]:.5f}, {h_vec[1]:.5f}, {h_vec[2]:.5f}]")
        print(f"  DAB: [{dab_vec[0]:.5f}, {dab_vec[1]:.5f}, {dab_vec[2]:.5f}]")
    
    def calculate_average_vectors(self):
        """
        Calculate average stain vectors from all added images (representing the entire lab's staining)
        """
        if len(self.all_vectors) == 0:
            raise ValueError("No vectors added yet!")
        
        # Extract all H and DAB vectors
        h_vectors = np.array([v['hematoxylin'] for v in self.all_vectors])
        dab_vectors = np.array([v['dab'] for v in self.all_vectors])
        
        # Calculate mean
        avg_h = np.mean(h_vectors, axis=0)
        avg_dab = np.mean(dab_vectors, axis=0)
        
        # Calculate standard deviation (variability across images)
        std_h = np.std(h_vectors, axis=0)
        std_dab = np.std(dab_vectors, axis=0)
        
        # Store
        self.hematoxylin_vector = avg_h
        self.dab_vector = avg_dab
        self.stain_matrix = np.array([avg_h, avg_dab])
        
        print("\n" + "="*70)
        print("AVERAGED STAIN VECTORS FROM QUPATH")
        print("="*70)
        print(f"Based on {len(self.all_vectors)} images\n")
        print(f"Hematoxylin: [{avg_h[0]:.5f}, {avg_h[1]:.5f}, {avg_h[2]:.5f}]")
        print(f"DAB:         [{avg_dab[0]:.5f}, {avg_dab[1]:.5f}, {avg_dab[2]:.5f}]")
        print(f"\nVariability (std dev):")
        print(f"Hematoxylin: [{std_h[0]:.5f}, {std_h[1]:.5f}, {std_h[2]:.5f}]")
        print(f"DAB:         [{std_dab[0]:.5f}, {std_dab[1]:.5f}, {std_dab[2]:.5f}]")
        print("="*70)
        
        # Check if variability is acceptable
        max_std = max(std_h.max(), std_dab.max())
        if max_std < 0.05:
            print("✓ Low variability - excellent consistency across images")
        elif max_std < 0.10:
            print("✓ Moderate variability - good consistency")
        else:
            print(" High variability")
        
        return self.stain_matrix
    
    def visualize_vector_consistency(self):
        """
        Visualize how consistent the vectors are across images
        """
        if len(self.all_vectors) < 2:
            print("Need at least 2 images to visualize consistency")
            return
        
        h_vectors = np.array([v['hematoxylin'] for v in self.all_vectors])
        dab_vectors = np.array([v['dab'] for v in self.all_vectors])
        names = [v['image_name'] for v in self.all_vectors]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot each RGB component
        components = ['Red', 'Green', 'Blue']
        colors = ['red', 'green', 'blue']
        
        for i, (comp, color) in enumerate(zip(components, colors)):
            # Hematoxylin
            ax = axes[0, i]
            h_vals = h_vectors[:, i]
            ax.scatter(range(len(h_vals)), h_vals, c=color, s=100, alpha=0.6)
            ax.axhline(h_vals.mean(), color='black', linestyle='--', 
                      label=f'Mean: {h_vals.mean():.4f}')
            ax.set_ylabel('Vector Component Value')
            ax.set_title(f'Hematoxylin - {comp} Component')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # DAB
            ax = axes[1, i]
            dab_vals = dab_vectors[:, i]
            ax.scatter(range(len(dab_vals)), dab_vals, c=color, s=100, alpha=0.6)
            ax.axhline(dab_vals.mean(), color='black', linestyle='--',
                      label=f'Mean: {dab_vals.mean():.4f}')
            ax.set_ylabel('Vector Component Value')
            ax.set_title(f'DAB - {comp} Component')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Stain Vector Consistency Across Images', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.show()
    
    def deconvolve_image(self, image):
        # 1. Convert to Optical Density (OD)
        img_float = image.astype(np.float64) / 255.0
        img_od = -np.log(img_float + 1e-6)
    
        # 2. Construct the 3x3 Matrix properly
        # We need a 3rd 'Residual' vector that is perpendicular to H and DAB
        h_v = self.hematoxylin_vector
        d_v = self.dab_vector
        res_v = np.cross(h_v, d_v) 
        res_v = res_v / np.linalg.norm(res_v)
    
        stain_matrix = np.array([h_v, d_v, res_v]) 
    
        # 3. Matrix Inversion
        # This is what creates the separation instead of just a color swap.
        inverse_matrix = np.linalg.inv(stain_matrix)
    
        # 4. Multiply OD by the INVERSE matrix
        # This subtracts the 'contribution' of one stain from the other.
        deconvolved = img_od @ inverse_matrix
    
        # Channel 0 is H, Channel 1 is DAB
        h_channel = np.clip(deconvolved[:, :, 0], 0, None)
        dab_channel = np.clip(deconvolved[:, :, 1], 0, None)
    
        return h_channel, dab_channel
    

    def test_deconvolution(self, image_path):

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Perform deconvolution
        h, dab = self.deconvolve_image(img_rgb)
        
        # 2. RECONSTRUCT RGB IMAGES
        # Use the Beer-Lambert law: Intensity = 255 * exp(-OD * Vector)
        # Add a newaxis so the 2D channel can be multiplied by the 1D [R, G, B] vector
        h_rgb_vis = 255 * np.exp(-h[:, :, np.newaxis] * self.hematoxylin_vector)
        dab_rgb_vis = 255 * np.exp(-dab[:, :, np.newaxis] * self.dab_vector)
        
        # 3. Finalize for display
        h_rgb_vis = np.clip(h_rgb_vis, 0, 255).astype(np.uint8)
        dab_rgb_vis = np.clip(dab_rgb_vis, 0, 255).astype(np.uint8)

        # Calculate correlation on raw OD data
        correlation = np.corrcoef(h.flatten(), dab.flatten())[0, 1]
        
        # Visualize
        fig = plt.figure(figsize=(16, 10))
        
        # Row 1: Original and RGB-reconstructed channels
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(img_rgb)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(h_rgb_vis) 
        ax2.set_title('Hematoxylin Only\n(Nuclei Reconstructed)', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3 = plt.subplot(2, 3, 3)
        ax3.imshow(dab_rgb_vis) 
        ax3.set_title('DAB Only\n(Protein Reconstructed)', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Row 2 
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(h.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Hematoxylin Intensity (OD)')
        ax4.set_title('Hematoxylin Distribution')
        
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(dab.flatten(), bins=50, color='brown', alpha=0.7, edgecolor='black')
        ax5.set_xlabel('DAB Intensity (OD)')
        ax5.set_title('DAB Distribution')
        
        ax6 = plt.subplot(2, 3, 6)
        sample_indices = np.random.choice(h.size, min(10000, h.size), replace=False)
        ax6.scatter(h.flatten()[sample_indices], dab.flatten()[sample_indices],
                   alpha=0.2, s=1, c='black')
        ax6.set_title(f'Correlation: {correlation:.3f}\n(L-shape = Excellent)')
        
        plt.tight_layout()
        plt.show()

        return h, dab
    
    def visualize_vector_consistency(self):
        """
        Visualize how consistent vectors are across images
        """
        if len(self.all_vectors) < 2:
            print("Need at least 2 images to visualize consistency")
            return
        
        h_vectors = np.array([v['hematoxylin'] for v in self.all_vectors])
        dab_vectors = np.array([v['dab'] for v in self.all_vectors])
        names = [v['image_name'] for v in self.all_vectors]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        components = ['Red', 'Green', 'Blue']
        colors = ['red', 'green', 'blue']
        
        for i, (comp, color) in enumerate(zip(components, colors)):
            # Hematoxylin
            ax = axes[0, i]
            h_vals = h_vectors[:, i]
            ax.scatter(range(len(h_vals)), h_vals, c=color, s=100, alpha=0.6)
            ax.axhline(h_vals.mean(), color='black', linestyle='--', linewidth=2,
                      label=f'Mean: {h_vals.mean():.4f}')
            ax.set_ylabel('Vector Component', fontsize=10)
            ax.set_title(f'Hematoxylin - {comp}', fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # DAB
            ax = axes[1, i]
            dab_vals = dab_vectors[:, i]
            ax.scatter(range(len(dab_vals)), dab_vals, c=color, s=100, alpha=0.6)
            ax.axhline(dab_vals.mean(), color='black', linestyle='--', linewidth=2,
                      label=f'Mean: {dab_vals.mean():.4f}')
            ax.set_ylabel('Vector Component', fontsize=10)
            ax.set_title(f'DAB - {comp}', fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Stain Vector Consistency Across Images', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def save_vectors(self, filename='qupath_stain_vectors.txt'):
        """
        Save averaged vectors to file
        """
        if self.stain_matrix is None:
            raise ValueError("Calculate average vectors first!")
        
        with open(filename, 'w') as f:
            f.write("# Stain Vectors from QuPath Analysis\n")
            f.write(f"# Averaged from {len(self.all_vectors)} images\n")
            f.write("# These are NORMALIZED OPTICAL DENSITY vectors\n")
            f.write("# Format: [R, G, B]\n\n")
            
            f.write(f"Hematoxylin: [{self.hematoxylin_vector[0]:.6f}, "
                   f"{self.hematoxylin_vector[1]:.6f}, "
                   f"{self.hematoxylin_vector[2]:.6f}]\n")
            f.write(f"DAB: [{self.dab_vector[0]:.6f}, "
                   f"{self.dab_vector[1]:.6f}, "
                   f"{self.dab_vector[2]:.6f}]\n")
            
            f.write("\n# Individual image vectors:\n")
            for v in self.all_vectors:
                f.write(f"\n{v['image_name']}:\n")
                f.write(f"  H:   [{v['hematoxylin'][0]:.6f}, "
                       f"{v['hematoxylin'][1]:.6f}, "
                       f"{v['hematoxylin'][2]:.6f}]\n")
                f.write(f"  DAB: [{v['dab'][0]:.6f}, "
                       f"{v['dab'][1]:.6f}, "
                       f"{v['dab'][2]:.6f}]\n")
        
        print(f"\n✓ Vectors saved to: {filename}")
    
    def load_vectors(self, filename):
        """
        Load previously saved vectors
        """
        vectors = {}
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('Hematoxylin:'):
                    vec_str = line.split('[')[1].split(']')[0]
                    vectors['h'] = np.array([float(x) for x in vec_str.split(',')])
                elif line.startswith('DAB:'):
                    vec_str = line.split('[')[1].split(']')[0]
                    vectors['dab'] = np.array([float(x) for x in vec_str.split(',')])
        
        self.hematoxylin_vector = vectors['h']
        self.dab_vector = vectors['dab']
        self.stain_matrix = np.array([vectors['h'], vectors['dab']])
        
        print("Stain vectors loaded successfully!")
        print(f"  H:   [{self.hematoxylin_vector[0]:.5f}, {self.hematoxylin_vector[1]:.5f}, {self.hematoxylin_vector[2]:.5f}]")
        print(f"  DAB: [{self.dab_vector[0]:.5f}, {self.dab_vector[1]:.5f}, {self.dab_vector[2]:.5f}]")
        
        return self.stain_matrix
    

class DABQuantifier:
    """
    Optimized DAB quantification for batch processing.
    """
    
    def __init__(self, reference_background=None, reference_std=None):
        self.reference_background = reference_background
        self.reference_std = reference_std
        self.dab_vector = np.array([0.368, 0.597, 0.706])
    
    def calibrate_from_background(self, background_images, visualize=True):
        """Calibrate background"""
        all_od_values = []
        image_stats = []
        
        print("="*60)
        print("CALIBRATING BACKGROUND FROM WHITE MATTER")
        print("="*60)
        
        for img_path in background_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_od = -np.log((img_rgb.astype(np.float64) / 255.0) + 1e-6) #OD conversion
            dab_od = np.dot(img_od, self.dab_vector)
            
            all_od_values.extend(dab_od.flatten()[::10])  # Subsample (every 10th pixel for memory)
            
            image_stats.append({
                'filename': Path(img_path).name,
                'median': np.median(dab_od),
                'mean': np.mean(dab_od)
            })
            
            print(f"{Path(img_path).name}: Median={np.median(dab_od):.4f}")
        
        if len(all_od_values) == 0:
            print("ERROR: No calibration images found.")
            return None, None
        
        all_od_values = np.array(all_od_values)
        
        # Robust statistics
        self.reference_background = np.median(all_od_values) #robust percentile
        q75 = np.percentile(all_od_values, 75)
        q25 = np.percentile(all_od_values, 25)
        self.reference_std = (q75 - q25) / 1.349
        
        print(f"\nBackground: {self.reference_background:.4f}")
        print(f"Robust SD: {self.reference_std:.4f}")
        
        if visualize:
            self._plot_calibration(all_od_values, image_stats)
        
        return self.reference_background, self.reference_std
    
    def _plot_calibration(self, all_od_values, image_stats):
        """Simplified calibration visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Histogram
        axes[0].hist(all_od_values, bins=200, alpha=0.7, color='brown', density=True)
        axes[0].axvline(self.reference_background, color='red', linestyle='--', 
                       linewidth=2, label=f'Median: {self.reference_background:.4f}')
        axes[0].set_xlabel('DAB OD')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Background Distribution')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Cumulative
        sorted_od = np.sort(all_od_values)
        cumulative = np.arange(1, len(sorted_od) + 1) / len(sorted_od) * 100
        axes[1].plot(sorted_od, cumulative, color='brown', linewidth=2)
        axes[1].axvline(self.reference_background, color='red', linestyle='--', linewidth=2)
        axes[1].axhline(50, color='gray', linestyle=':', alpha=0.5)
        axes[1].set_xlabel('DAB OD')
        axes[1].set_ylabel('Cumulative %')
        axes[1].set_title('Cumulative Distribution')
        axes[1].grid(alpha=0.3)
        
        # Per-image
        df_stats = pd.DataFrame(image_stats)
        axes[2].scatter(range(len(df_stats)), df_stats['median'], 
                       color='green', s=80, alpha=0.7, edgecolor='black')
        axes[2].axhline(self.reference_background, color='red', linestyle='--', linewidth=2)
        axes[2].set_xlabel('Image Index')
        axes[2].set_ylabel('Median DAB OD')
        axes[2].set_title('Per-Image Background')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def quantify_image(self, image_path, threshold_method='fixed_adaptive', 
                      threshold_param=1.5, tissue_mask_threshold=245):
        """
        Quantify DAB in a single image.
        
        Args:
            image_path: Path to image
            threshold_method: 
                'fixed' - k × background_std (param = k, default 1.5)
                'fixed_adaptive' - Combines global + local stats (param = global_weight, default 1.5)
                'triangle' - Triangle algorithm (param ignored)
                'percentile' - Based on percentile range (param = sensitivity, default 0.3)
                'multiscale' - Multi-scale combination (param = conservativeness, default 1.5)
            threshold_param: Parameter for threshold method
            tissue_mask_threshold: Tissue detection (default 245)
        
        Returns:
            Dictionary with essential metrics only
        """
        if self.reference_background is None:
            raise ValueError("Must calibrate background first!")
        
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Tissue mask
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            tissue_mask = gray < tissue_mask_threshold
            
            if not np.any(tissue_mask):
                return None
            
            # DAB extraction
            img_od = -np.log((img_rgb.astype(np.float32) / 255.0) + 1e-6)
            dab_od = np.dot(img_od, self.dab_vector)
            
            # Background subtraction
            dab_signal = dab_od - self.reference_background
            dab_signal = np.clip(dab_signal, 0, None)
            
            # Get tissue pixels
            dab_tissue = dab_signal[tissue_mask]
            
            if len(dab_tissue) == 0 or np.max(dab_tissue) == 0:
                return None
            
            # Calculate threshold
            threshold = self._calculate_threshold(
                dab_tissue, threshold_method, threshold_param
            )
            
            # Binary mask
            dab_positive = (dab_signal > threshold) & tissue_mask
            
            # Calculate metrics
            tissue_pixels = np.sum(tissue_mask)
            positive_pixels = np.sum(dab_positive)
            
            if positive_pixels > 0:
                positive_intensity_mean = np.mean(dab_signal[dab_positive])
                positive_intensity_sum = np.sum(dab_signal[dab_positive])
            else:
                positive_intensity_mean = 0.0
                positive_intensity_sum = 0.0
            
            # Essential metrics only
            results = {                
                'filename': Path(image_path).name,
                
                # RAW COUNTS
                'positive_pixels': int(positive_pixels),
                'tissue_pixels': int(tissue_pixels),
                
                # AREA METRICS (normalized by tissue)
                'area_percent': round((positive_pixels / tissue_pixels * 100), 3),
                
                # INTENSITY METRICS (positive regions only)
                'mean_positive_intensity': round(positive_intensity_mean, 4),
                'total_positive_dab': round(positive_intensity_sum, 2),
                
                # NORMALIZED INTENSITY
                'dab_density': round((positive_intensity_sum / tissue_pixels), 6),
                
                # THRESHOLD INFO
                'threshold_value': round(threshold, 4),
                'threshold_method': threshold_method,
                
                # QC
                'tissue_coverage_percent': round((tissue_pixels / dab_signal.size * 100), 2),
                'max_dab_signal': round(np.max(dab_tissue), 4)
            }
            
            return results
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def _calculate_threshold(self, dab_tissue, method, param):
        """Calculate threshold using specified method."""
        
        if method == 'fixed':
            # Simple: k × SD above background
            threshold = param * self.reference_std
        
        elif method == 'fixed_adaptive':
            # Combines global background stats with local tissue stats
            global_thresh = param * self.reference_std
            local_mean = np.mean(dab_tissue)
            local_std = np.std(dab_tissue)
            
            # Only use local if tissue has reasonable signal
            if local_mean > 2 * self.reference_std:
                local_thresh = local_mean + 0.5 * local_std
                # Weight toward global (more stable)
                threshold = 0.7 * global_thresh + 0.3 * local_thresh
            else:
                threshold = global_thresh
        
        elif method == 'triangle':
            # Triangle algorithm - good for skewed distributions
            dab_norm = (dab_tissue / np.max(dab_tissue) * 255).astype(np.uint8)
            try:
                threshold = filters.threshold_triangle(dab_norm)
                threshold = (threshold / 255.0) * np.max(dab_tissue)
            except:
                threshold = param * self.reference_std
        
        elif method == 'percentile':
            # Percentile-based: find pixels in top percentage
            p98 = np.percentile(dab_tissue, 98)
            p75 = np.percentile(dab_tissue, 75)
            # param controls sensitivity: 0.3 = moderate, 0.5 = sensitive
            threshold = p75 + param * (p98 - p75)
        
        elif method == 'multiscale':
            # Multi-scale: combines multiple threshold estimates
            # Good for heterogeneous staining
            
            # Method 1: Fixed
            t1 = param * self.reference_std
            
            # Method 2: Mean + SD
            t2 = np.mean(dab_tissue) + np.std(dab_tissue)
            
            # Method 3: Percentile
            t3 = np.percentile(dab_tissue, 75) + 0.3 * (
                np.percentile(dab_tissue, 98) - np.percentile(dab_tissue, 75)
            )
            
            # Weight toward more conservative (higher) threshold
            threshold = np.median([t1, t2, t3])
        
        else:
            # Default fallback
            threshold = 1.5 * self.reference_std
        
        return threshold
    
    def test_thresholds(self, test_images, tissue_mask_threshold=245, save_fig=True):
        """
        Test different threshold methods on representative images.
        
        Args:
            test_images: List of paths [pale_image, intense_image]
            tissue_mask_threshold: Tissue detection threshold
            save_fig: Save comparison figure
        """
        if self.reference_background is None:
            raise ValueError("Must calibrate background first!")
        
        methods_to_test = [
            ('fixed', 1.0, 'Fixed: 1.0σ'),
            ('fixed', 1.5, 'Fixed: 1.5σ'),
            ('fixed', 2.0, 'Fixed: 2.0σ'),
            ('fixed_adaptive', 3.0, 'Adaptive Fixed'),
            ('triangle', None, 'Triangle'),
            ('percentile', 0.3, 'Percentile (0.3)'),
            ('multiscale', 1.5, 'Multi-scale')
        ]
        
        results_table = []
        
        print("\n" + "="*80)
        print("THRESHOLD METHOD COMPARISON")
        print("="*80)
        
        for img_path in test_images:
            print(f"\nImage: {Path(img_path).name}")
            print("-"*80)
            print(f"{'Method':<20} {'Threshold':<12} {'Area %':<10} {'Mean Int':<12} {'DAB Density'}")
            print("-"*80)
            
            for method, param, label in methods_to_test:
                result = self.quantify_image(
                    img_path,
                    threshold_method=method,
                    threshold_param=param if param else 1.5,
                    tissue_mask_threshold=tissue_mask_threshold
                )
                
                if result:
                    print(f"{label:<20} {result['threshold_value']:<12.4f} "
                          f"{result['area_percent']:<10.2f} "
                          f"{result['mean_positive_intensity']:<12.4f} "
                          f"{result['dab_density']:.6f}")
                    
                    results_table.append({
                        'image': Path(img_path).name,
                        'method': label,
                        **result
                    })
        
        # Create comparison visualization
        if save_fig and len(test_images) == 2:
            self._plot_threshold_comparison(test_images, methods_to_test, tissue_mask_threshold)
        
        return pd.DataFrame(results_table)
    
    def _plot_threshold_comparison(self, test_images, methods, tissue_threshold):
        """Create side-by-side comparison of threshold methods."""
        fig, axes = plt.subplots(len(test_images), len(methods) + 1, 
                                figsize=(4*(len(methods)+1), 5*len(test_images)))
        
        if len(test_images) == 1:
            axes = axes.reshape(1, -1)
        
        for row, img_path in enumerate(test_images):
            # Load image
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Original
            axes[row, 0].imshow(img_rgb)
            axes[row, 0].set_title(f'Original\n{Path(img_path).name}', fontsize=10)
            axes[row, 0].axis('off')
            
            # Each method
            for col, (method, param, label) in enumerate(methods, 1):
                result = self.quantify_image(
                    img_path,
                    threshold_method=method,
                    threshold_param=param if param else 1.5,
                    tissue_mask_threshold=tissue_threshold
                )
                
                if result:
                    # Create overlay
                    overlay = img_rgb.copy()
                    # Recreate mask for visualization
                    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                    tissue_mask = gray < tissue_threshold
                    img_od = -np.log((img_rgb.astype(np.float32) / 255.0) + 1e-6)
                    dab_od = np.dot(img_od, self.dab_vector)
                    dab_signal = np.clip(dab_od - self.reference_background, 0, None)
                    
                    threshold = result['threshold_value']
                    dab_positive = (dab_signal > threshold) & tissue_mask
                    
                    overlay[dab_positive] = [255, 0, 0]
                    
                    axes[row, col].imshow(overlay)
                    title = (f'{label}\nThresh: {threshold:.3f}\n'
                            f'Area: {result["area_percent"]:.1f}%\n'
                            f'Mean: {result["mean_positive_intensity"]:.4f}')
                    axes[row, col].set_title(title, fontsize=9)
                    axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('threshold_comparison.png', dpi=200, bbox_inches='tight')
        plt.show()
        print("\n✓ Comparison saved to: threshold_comparison.png")
    
    def batch_process(self, image_folder, output_csv='results.csv',
                     output_masks_folder=None, threshold_method='fixed_adaptive',
                     threshold_param=1.5, tissue_mask_threshold=245):
        """
        Optimized batch processing - saves only essential metrics.
        
        Args:
            image_folder: Folder with images
            output_csv: Output CSV file
            output_masks_folder: Optional folder to save masks
            threshold_method: Threshold algorithm
            threshold_param: Parameter for threshold
            tissue_mask_threshold: Tissue detection threshold
        
        Returns:
            DataFrame with results
        """
        image_folder = Path(image_folder)
        image_files = list(image_folder.glob('*.png')) + \
                      list(image_folder.glob('*.jpg')) + \
                      list(image_folder.glob('*.tif*'))
        
        if output_masks_folder:
            output_masks_folder = Path(output_masks_folder)
            output_masks_folder.mkdir(parents=True, exist_ok=True)
        
        results_list = []
        
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING: {len(image_files)} images")
        print(f"Method: {threshold_method}, Param: {threshold_param}")
        print(f"{'='*70}\n")
        
        for i, img_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] {img_path.name}...", end=' ')
            
            result = self.quantify_image(
                img_path,
                threshold_method=threshold_method,
                threshold_param=threshold_param,
                tissue_mask_threshold=tissue_mask_threshold
            )
            
            if result:
                # Save mask if requested
                if output_masks_folder:
                    self._save_mask(img_path, result, output_masks_folder, 
                                   threshold_param, tissue_mask_threshold)
                
                results_list.append(result)
                print(f"✓ (Area: {result['area_percent']:.2f}%, DAB: {result['mean_positive_intensity']:.4f})")
            else:
                print("✗ Failed")
        
        df = pd.DataFrame(results_list)
        df.to_csv(output_csv, index=False)
        
        print(f"\n{'='*70}")
        print(f"✓ Complete! Processed: {len(results_list)}/{len(image_files)}")
        print(f"  Results: {output_csv}")
        if output_masks_folder:
            print(f"  Masks: {output_masks_folder}")
        print(f"{'='*70}\n")
        
        return df
    
    def _save_mask(self, img_path, result, output_folder, threshold_param, tissue_threshold):
        """Save visualization mask for QC."""
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Recreate mask
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        tissue_mask = gray < tissue_threshold
        img_od = -np.log((img_rgb.astype(np.float32) / 255.0) + 1e-6)
        dab_od = np.dot(img_od, self.dab_vector)
        dab_signal = np.clip(dab_od - self.reference_background, 0, None)
        
        threshold = result['threshold_value']
        dab_positive = (dab_signal > threshold) & tissue_mask
        
        # Create RGB mask: white=tissue, red=positive
        mask_vis = np.zeros((*dab_positive.shape, 3), dtype=np.uint8)
        mask_vis[tissue_mask] = [255, 255, 255]
        mask_vis[dab_positive] = [255, 0, 0]
        
        mask_filename = output_folder / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(mask_filename), cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))
    
    def save_calibration(self, filepath='background_calibration.npz'):
        """Save calibration."""
        np.savez(filepath, 
                 reference_background=self.reference_background,
                 reference_std=self.reference_std)
        print(f"✓ Calibration saved: {filepath}")
    
    def load_calibration(self, filepath='background_calibration.npz'):
        """Load calibration."""
        data = np.load(filepath)
        self.reference_background = float(data['reference_background'])
        self.reference_std = float(data['reference_std'])
        print(f"✓ Calibration loaded: {filepath}")
        print(f"  Background: {self.reference_background:.4f}")
        print(f"  SD: {self.reference_std:.4f}")

