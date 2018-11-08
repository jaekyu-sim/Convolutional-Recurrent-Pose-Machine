# Convolutional-Recurrent-Pose-Machine
This project is for detecting human body in any situation. I planned this project because CPM(Convolutional Pose Machine) can't detect human body when human is behind abstacle.


![abstacle image2](/images/15.jpg)
![abstacle image3](/images/0.jpg)

These above images are human images who is dancing with white rectangular abstacle.(opencv rectangular)

When this image is put in CPM Network, the output is not correct because CPM only can detect human body on one image.

So, I suggest network that has recurrent shape.

