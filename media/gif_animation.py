from PIL import Image

imgs = []

imgs.append(Image.open('./chess1.png').convert('RGB'))
imgs.append(Image.open('./chess2.png').convert('RGB'))

imgs[0].save('chess.gif', save_all=True, append_images=imgs[1:],
             optimize=False, duration=1000, loop=0)
