import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow_hub as hub

def load_and_process_image(image_path, target_size=(512, 512)):  # Changed to 224x224
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = tf.cast(img, tf.float32) / 255.0
    return img

# Load VGG19
vgg = VGG19(weights='imagenet', include_top=False)
print("VGG19 model loaded successfully!")

# Content and style layers
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']



@tf.function
# Use TensorFlow Hub's pre-trained fast style transfer
def fast_style_transfer(content_path, style_path, style_blend_weight=1.0):
    # Load TF Hub model
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    
    content_image = load_and_process_image(content_path, (512, 512))
    style_image = load_and_process_image(style_path, (256, 256))
    
    # Apply style transfer (remove tf.constant)
    stylized_image = hub_model(content_image, style_image)[0]
    final_result = style_blend_weight * stylized_image + (1 - style_blend_weight) * content_image
    
    
    return final_result


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor

def save_image(tensor, filename):
    image = tensor_to_image(tensor)
    plt.figure(figsize=(12, 4))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(f'outputs/{filename}', bbox_inches='tight', dpi=150)
    plt.show()

# Update your main function:
if __name__ == "__main__":
    result = fast_style_transfer('images/content.jpg', 'images/style.jpg', style_blend_weight=0.5)
    save_image(result, 'stylized_output.png')
    print("Fast style transfer complete!")