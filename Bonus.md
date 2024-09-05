# Bonus Tasks - CNN with Laplacian and Prewitt Filters
In addition to the primary Sobel filter task, few bonus tasks were also implemented. These tasks explore the application of other classical image processing filters, like Prewitt and Laplacian, using CNNs. Furthermore, the project includes visualizations of the CNN layer activations too.

The complete code can be found in `bonus-hackathon.ipynb` file.


## Bonus Task 1: Training the model for Laplacian and Prewitt Filtered Data
The Laplacian filter detects edges by calculating the second derivative of the image. Unlike Sobel, which detect edges based on gradients, the Laplacian filter captures more complex edge details.

The Prewitt filter is another edge detection operator, similar to the Sobel filter, but with a simpler kernel. The goal was to train a CNN to approximate the Prewitt filter in a manner similar to the Sobel filter task.

## Bonus Task 2: Visualizing CNN Layer Activations for Different Filters

The activations of the CNN layers were visualized to understand the features being learned at different stages of the Prewitt model.


## Conclusion
The U-Net model performed well across Sobel and Laplacian-filtered datasets, demonstrating robust edge detection and feature extraction capabilities. The Prewitt-filtered dataset, however, showed lower performance metrics and SSIM, which highlights the need for further exploration.

The bonus tasks provided a deeper exploration into how CNNs can generalize various classical image filters, not just Sobel. By extending the project to include Prewitt and Laplacian filters, I gained valuable insights into how CNNs can be applied to approximate a variety of traditional image processing methods.
