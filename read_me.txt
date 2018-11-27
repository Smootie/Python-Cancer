Assignment 6ec
Tom Smoot
TES2135

Since I used the scipy KDtree function in assignment 6b, I'm reusing all of the code for the extra credit. The function has an additional parameter to specify how to find distance. In 6b, I set that to euclidian distance, but here I'm passing a variable to change that. 

This asssignment should work as expected. I had the output consist of stating the accuracy of validating the synthetic data and the real-world data. 

I didn't see in the instructions the requirement of stating whether the test data for the real world dataset should be shown, even though I wanted to show whether tested cells were malignant or benign. 

I chose to roll the real-world data to get the labels at the end of the rows, because that was how I built my synthetic data. I also chose to replace the 'malignant' identifier as 1 (true) if it matters. 

I was surprised by the accuracy, but read in the data instructions that they used a simliar cross-validation procedure to achieve over 96% accuracy, so I am accepting of my results. 