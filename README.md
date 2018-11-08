# Convolutional-Recurrent-Pose-Machine
This project is for detecting human body in any situation. I planned this project because CPM(Convolutional Pose Machine) can't detect human body when human is behind abstacle.


![abstacle image2](/images/15.jpg)
![abstacle image3](/images/0.jpg)

These above images are human images who is dancing with white rectangular abstacle.(opencv rectangular)

When this image is put in CPM Network, the output is not correct because CPM only can detect human body on one image.

So, I suggest network that has recurrent shape.

![network](/images/network.PNG)

I applies recurrent network to CPM to detect skeleton position behind abstacle.

The output is here.

![output](/images/output.PNG)

I put a video to test the performance of my network comparing to CPM.

Input video is consisted with 30 frame size. 1st\~13rd frames are not applied abstacle, and 14th\~30th frames are applied abstacle.

I'll compare performance by 15th frame images(abstacle image)

the (a) is images which doesn't apply any abstacle.
I put (a) to CPM, and I got (b)
the output(b) is very clear.

the (c) is images which apply abstacle to head.
I put (c) to CPM, and I got (d)
The output(d) is not clear comparing to (b). There is ambigous head joint. 
But, the output(e) is clear comparing to (d) and it is almost same comparing to (b)
Because, CPM can detect skeleton data in one images, not determining previous image.
If the network determines previous image, network can detect human body skeleton data.

