# Run MNIST CNN over aws spot instance
Our example represent mnist dataset using convolutional neural network.
The MNIST dataset is an image dataset of handwritten digits. It has has 60,000 training images and 10,000 test images, each of which are grayscale 28 x 28 sized images. 

#### Train deep learning modules on ec2 cloud platform:
In order to develop and train deep learning NN models over cloud platforms you don't need much, you simply start your OD instance with your favorite GPU\CPU platform and aws custom ml ami, clone your code to the server and start running. 
`BUT` 
training NN takes hours or even days depending on the complexity of the network and the dataset size and aws gpu servers are not that cheap as on demand instances (consistent servers)

Training NN over spot instances instead of on demand instances: 
Cons:
The main issue of using spot instances while training your NN is the persistency issue, since aws gave us the server with significant discount they can interrupt it as well with just a short notice, Therefore, it’s not recommended for time-sensitive workloads.
Instance termination can cause data loss if the training progress is not saved properly. 
Persistency spot instances loses their data and infrastructure structure on interrupti

#### Training NN over spot instances: Principles 

* `Decouple: compute, storage and code, and keep the compute instance stateless`: This enables easy recovery and training state restore when an instance is terminated and replaced.
* `Use a dedicated volume for datasets`, training progress (checkpoints) and logs. This volume should be persistent and not be affected by instance termination.
* `Use a version control system` (e.g. Git) for training code. 
* `Minimize code changes to the training script` This ensures that the training script can be developed independently and backup and snapshot operations are performed outside of the training code.

#### Prerequisites:
* Active  spotinst account connected to AWS. https://api.spotinst.com/getting-started-with-elastigroup/
* Github repository with your training code.

#### Explanations
* Checkpoints: Checkpoints capture the exact value of all parameters used by a model. Checkpoints do not contain any description of the computation defined by the model and thus are typically only useful when source code that will use the saved parameter values is available. Checkpoints are using for continues training of your NN, we will use check points in our projects to avoid data loss on interruptions.
* Callbacks: A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training. 


