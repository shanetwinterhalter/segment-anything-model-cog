# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from cv2 import imread, imwrite, IMWRITE_PNG_COMPRESSION
from numpy import array as nparray
from numpy import uint8
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        self.sam = sam_model_registry['vit_h'](
            checkpoint=sam_checkpoint).to("cuda")

    def predict(
        self,
        image: Path = Input(description="Input image")
    ) -> list[Path]:
        """Run a single prediction on the model"""
        img = imread(str(image))
        img_array = nparray(img)
        mask_generator = SamAutomaticMaskGenerator(self.sam)
        masks = mask_generator.generate(img_array)
        mask_paths = [] * len(masks)
        for idx, mask in enumerate(masks):
            filename = "mask" + str(idx) + ".png"
            mask_segmentation = (mask["segmentation"] * 255).astype(uint8)
            imwrite(filename, mask_segmentation, [IMWRITE_PNG_COMPRESSION, 9] )
            mask_paths.append(Path(filename))
        return mask_paths
