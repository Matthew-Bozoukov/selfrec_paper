### Utility script to combine images into larger figure
 
from PIL import Image

image_files = ['self_min_other_positionXlayer.png', 'self_positionXlayer.png', 'other_positionXlayer.png']
nrow=1
ncol=3

images = [Image.open(x) for x in image_files]

# Calculate maximum width and height of images
max_width = max(img.width for img in images)
max_height = max(img.height for img in images)

new_image = Image.new('RGB', (max_width * ncol, max_height * nrow))

# Calculate positions to center each image in its grid cell
positions = []
for i in range(nrow):
    for j in range(ncol):
        positions.append(((max_width - images[i*ncol+j].width) // 2 + j*max_width, (max_height - images[i*ncol+j].height) // 2 + i*max_height))
#positions = [
#    ((max_width - images[0].width) // 2, (max_height - images[0].height) // 2),
#    (max_width + (max_width - images[1].width) // 2, (max_height - images[1].height) // 2),
#    ((max_width - images[2].width) // 2, max_height + (max_height - images[2].height) // 2),
#    (max_width + (max_width - images[3].width) // 2, max_height + (max_height - images[3].height) // 2)
#]

# Paste images into the new image
for img, pos in zip(images, positions):
    new_image.paste(img, pos)

new_image.save('combined_positionwise_heatmaps.png')
