from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def success(input_image_path,
				   output_image_path,
				   watermark_image_path,
				   text, pos):
	photo = Image.open(input_image_path)
	watermark = Image.open(watermark_image_path)
	watermark = watermark.resize((300,50))

	# make the image editable
	drawing = ImageDraw.Draw(photo)
	photo.paste(watermark, pos)
	color = (23, 155, 115)
	font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
	drawing.text(pos, text, fill=color, font=font)
	photo.show()
	photo.save(output_image_path)

def failure(input_image_path,
				   output_image_path,
				   watermark_image_path,
				   text, pos):
	photo = Image.open(input_image_path)
	watermark = Image.open(watermark_image_path)
	watermark = watermark.resize((350,50))

	# make the image editable
	drawing = ImageDraw.Draw(photo)
	photo.paste(watermark, pos)
	color = (192, 192, 192)
	font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
	drawing.text(pos, text, fill=color, font=font)
	photo.show()
	photo.save(output_image_path)




#if __name__ == '__main__':
#	img = 'test.jpg'
#
#	watermark(img, 'pic_watermarked.jpg',
#				   'background.jpg',
#				   text='Pig detected',
#				   pos=(0, 0))