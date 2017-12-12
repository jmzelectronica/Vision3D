from PIL import Image

im = Image.open("'/home/lab-10/Documents/Dataset_cactus_update/Negative_images/*.png'")
rgb_im = im.convert('RGB')
rgb_im.save('*.jpg')
