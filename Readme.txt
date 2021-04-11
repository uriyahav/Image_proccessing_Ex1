# python verision: 3.8
#platforms verisions : matplotlib 3.3.4 cv2 4.5.1.48 numpy 1.19.5
files submitted:
1) ex1_utils.py- main python file with function of  image proccessing such as quantizaztion , gamma correction , transform between YIQ to RGB etc.
short description of the functions:
*imReadAndConvert- function to read image and convert & return the imagee with the requested representation - rgb or gray scale, to convert i used cv2.cvtcolor.
*imDisplay -  this function used imReadAndConvert and represent the photo with the requested representation - rgb or gray scale using plot(matplotlib).
*transformRGB2YIQ - given a photo with Red Green Blue represntation and transform it to YIQ color space. we work only on the Y channel. for doing it we used np array to yiq transfor as given in the class. return the image represented in yiq color space
*transformYIQ2RGB- opposite transform for the function above.
*hsitogramEqualize- function to preforn histogrm equalization of a given photo(rgb or gray scale)
output - equalized image , histogram of the original photo, and the quantizized.in case of rgb  we work only on the y channe -   collect the information in the y chnnnel and transform in thhe end back to rgb.
*quantizeImage- in this function we quantized an image(rgb/gray scale) into nQuant colors.
The function returns a list of the quantized image(image for each iteration), list of MSE(one for each iteration).
to preform the quantization i add several function:
init_z- function to initialize z based on cumsum of image histogram.
we split the pixels into aprroximatly even k means.
finding_q-  for check and change our q array based on t and the histogram.
finding_z- to change z based on q array. 
quantizaztion - function to apply quantizaztion and return the quantized image.
after initialize z   and q we run nIter times and at each iteration we call find z , find q and quantizaztion, then calculate the error(mse we learned at class), and add the image and the error to lists.

2) gamma.py-in this file we perform gamma correction on an image with a given gamma(you can change it using the trackbar) and represent it.
short description of the functions:
gammaDisplay-this function we perform gamma correction on an image(rgb/gray scale) with a given gamma(you can change it using the trackbar) and represent it.
the gamma is between 0-2 , represent in the plot as number betwween 0-200.
the user can chaange the gamma and see how i effects the photo.
#testimg1.jpg- for equalization & testimg2.jpg to check the quantization & test rgb and grayscale.
on both of them i try rgb & gray scale.
the reason i choose this pictures  is because i searched on the internet for examples for quantization  & equalization and took similar photos that could check my function.
as you can see  in testimg1 we got beck 'better looking' picture , and in testimg2 you can see the quantization and how the rgb to gray scale works.

3) ex1_main.py- main file given for the task.

4) estimg1.jpg- for equalization 

5) testimg2.jpg to check the quantization & test rgb and grayscale.
