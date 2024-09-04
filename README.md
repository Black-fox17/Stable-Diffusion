# Stable Diffusion from Scratch: Learning Journey

Welcome to my journey of coding Stable Diffusion from scratch, inspired by Umar Jamil's video series. This README documents what I've learned along the way, breaking down some of the key concepts in a way that’s accessible yet detailed.

## Day 1:

### Understanding Stable Diffusion
On the first day, I dived into the basics of Stable Diffusion, a process that helps in generating high-quality images by iteratively refining a noisy image. The goal is to gradually transform this noise into something meaningful, like a clear image of a cat or a landscape.

### ELBO (Evidence Lower Bound)
I explored the mathematics behind ELBO, which stands for Evidence Lower Bound. ELBO is crucial in the context of variational inference—a method used to approximate complex probability distributions. In simpler terms, it helps us train models by providing a balance between the complexity of the model and its fit to the data.

The ELBO can be expressed as:
\[
\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x) \| p(z))
\]
Here’s what this formula means:
- The first term, \(\mathbb{E}_{q(z|x)}[\log p(x|z)]\), represents the likelihood of the data under the model.
- The second term, \(D_{\text{KL}}(q(z|x) \| p(z))\), is the Kullback-Leibler divergence, which measures how different the approximated distribution \(q(z|x)\) is from the true distribution \(p(z)\).

By maximizing the ELBO, we ensure that our model accurately captures the data while remaining as simple as possible.

### Importance of Statistics in GANs
Generative Adversarial Networks (GANs) are a fascinating type of model that generates new data similar to a given dataset. I learned how statistics play a vital role in GANs. A GAN consists of two parts:
1. **Generator:** Creates fake data.
2. **Discriminator:** Tries to distinguish between real  and fake data.

The generator improves by trying to fool the discriminator, while the discriminator gets better at spotting fakes. This adversarial process is statistically grounded, and understanding concepts like probability distributions and optimization techniques is key to mastering GANs.

### CLIP (Contrastive Language-Image Pretraining)
CLIP is a model that connects images with text, a fundamental task in AI. It’s trained on a vast dataset of images and their corresponding captions. CLIP’s role in GANs is to ensure that when you generate an image from text (like "a sunset over mountains"), the result accurately reflects the description.

### CFG (Classifier-Free Guidance)
Classifier-Free Guidance (CFG) is a technique that enhances the quality of generated images by guiding the process without relying on a predefined classifier. This method reduces bias in the output, ensuring that the generated images align more closely with the intended conditions. For example, if you want to generate images of animals, CFG helps the model produce diverse yet accurate images without falling into common biases.

### UNet Architecture
UNet is a powerful neural network architecture, particularly useful in image processing tasks like segmentation. Its unique structure, which consists of an encoder (to compress the input) and a decoder (to reconstruct the output), makes it ideal for tasks where preserving the original image's details is crucial.

In Stable Diffusion, UNet plays a central role in refining images by iteratively improving them, whether it’s transforming one image into another, generating images from text, or even inpainting (filling in missing parts of an image).

### Variational Autoencoders (VAE)
VAEs are a type of autoencoder that adds a probabilistic twist to the traditional autoencoder structure. They’re particularly useful in generating new data by sampling from a latent space—a compressed representation of the data. VAEs are preferred in Stable Diffusion because they allow for more controlled and diverse generation of images.

The VAE loss function combines two components:
1. **Reconstruction Loss:** Measures how well the VAE can reconstruct the original data from the latent space.
2. **KL Divergence:** Ensures that the latent space has a meaningful structure by comparing the learned distribution with a prior distribution.

The total loss is a balance between accurately recreating the input and maintaining a smooth, continuous latent space from which we can sample new data.

---

This README is a work in progress, and I'll continue to update it as I learn more. If you’re interested in the technical details or want to follow along, feel free to explore the code and resources linked here!
