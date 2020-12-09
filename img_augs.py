class ImgAugTransform_cars_trucks:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class ImgAugTransform_cats:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            iaa.Rotate(90),
            iaa.Rotate(0),
            iaa.Roate(180),
            iaa.Rotate(270)
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


transforms_cars = ImgAugTransform_cars_trucks()
transforms_cats=ImgAugTransform_cats()

dataset = torchvision.datasets.ImageFolder('PATH_DATA_SET/', transform=transforms)

for i,img in enumerate(dataset):

  ag=[]
  imgs = [np.asarray(img[0]) for _ in range(6)]
  if i>4977:
    ag=transforms_cats.augment_images(imgs)
  if i<=4977:
    ag = transforms_cars.augment_images(imgs)
  for j,m in enumerate(ag):
    if i<4528:
      plt.imsave("/content/aug1/0/"+str(i)+"_"+str(j)+".jpg",m)
    if 4528<i<4977:
      plt.imsave("/content/aug1/1/"+str(i)+"_"+str(j)+".jpg",m)
    if 4977<i:
      plt.imsave("/content/aug1/2/"+str(i)+"_"+str(j)+".jpg",m)


