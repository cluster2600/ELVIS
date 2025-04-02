import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ASCII Art for ELVIS
ELVIS_ASCII = r"""
 _______  _        __      __  _____   _____ 
|  ____| | |       \ \    / / |_   _| / ____|
| |__    | |        \ \  / /    | |  | (___  
|  __|   | |         \ \/ /     | |   \___ \ 
| |____  | |____      \  /     _| |_  ____) |
|______| |______|      \/     |_____||_____/ 
"""

# Create a new image with a black background
width, height = 800, 400
image = Image.new('RGB', (width, height), color='black')
draw = ImageDraw.Draw(image)

# Add the ASCII art as text
try:
    # Try to use a monospace font
    font = ImageFont.truetype("Courier", 36)
except:
    # Fallback to default font
    font = ImageFont.load_default()

# Draw the ASCII art in white
draw.text((50, 100), ELVIS_ASCII, fill='white', font=font)

# Add "Enhanced Leveraged Virtual Investment System" below
subtitle = "Enhanced Leveraged Virtual Investment System"
draw.text((150, 300), subtitle, fill='white', font=font)

# Save the image
image.save('images/elvis.png')
print("Image saved to images/elvis.png")
