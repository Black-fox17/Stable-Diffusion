Stable Diffusion from Scratch: Learning Journey
This repository documents my journey as I code Stable Diffusion from scratch, inspired by Umar Jamil's video series.

Day 1:

Understanding Stable Diffusion: I began by learning the foundational concepts of how Stable Diffusion works. This includes the mathematics behind the Evidence Lower Bound (ELBO), which is crucial for training models in a probabilistic framework. ELBO helps in optimizing the model to approximate complex distributions.

Importance of Statistics in GANs: I explored the significant role that statistical concepts play in Generative Adversarial Networks (GANs). GANs consist of two models, a generator and a discriminator, that compete against each other to produce more realistic data. Understanding the statistical underpinnings is essential for effectively training these networks.

CLIP (Contrastive Language-Image Pretraining): I delved into CLIP, a model that learns to associate images and text by training on large datasets. It plays a key role in text-to-image tasks within GANs, emphasizing the importance of Natural Language Processing (NLP) in generating contextually relevant images.

CFG (Classifier-Free Guidance): CFG is a technique used to steer the output of GANs towards desired characteristics, reducing bias and improving the alignment of the generated content with specified conditions.

UNet Architecture: The UNet, a neural network architecture commonly used in image processing tasks, is crucial in Stable Diffusion. I studied its structure and its role in tasks like image-to-image translation, text-to-image generation, and inpainting (filling in missing parts of images).

Variational Autoencoders (VAE): Finally, I learned about VAEs, a type of autoencoder preferred in Stable Diffusion for their ability to generate more diverse and realistic outputs by sampling from a latent space.

