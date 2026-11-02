import numpy as np

from fcgno.data.coefficients import COEFFICIENT_SAMPLERS
from fcgno.data.generate import make_dataset
from fcgno.experiments.settings import TABLE_KIND
from nfcg.data.coefficient_features import coefficient_contrast
from nfcg.data.random_fields import CONTRAST_DECADES


# Importing nfcg registers the high-contrast kinds, which use Diffusion training settings
def test_kinds_registered():
    for kind in CONTRAST_DECADES:
        assert kind in COEFFICIENT_SAMPLERS and TABLE_KIND[kind] == "diffusion"


# Generated coefficients are positive, differ between problems, and reach the requested contrast on the faces
def test_contrast_and_variety():
    for kind, decades in CONTRAST_DECADES.items():
        dataset = make_dataset(kind, 32, 3, seed=1, normalization="upstream")
        np.testing.assert_allclose(np.log10(coefficient_contrast(dataset)), decades, atol=0.1)
        assert dataset["face_x"].min() > 0 and not np.allclose(dataset["coefficient"][0], dataset["coefficient"][1])
