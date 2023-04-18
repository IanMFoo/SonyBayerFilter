## SONY Volumetric Inverse Photonic (VIP) Design Optimizer
Copyright © 2023, California Institute of Technology. All rights reserved.

Use in source and binary forms for nonexclusive, nonsublicenseable, commercial purposes with or without modification, is permitted provided that the following conditions are met:

* Use of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Use in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the software.
* Neither the name of the California Institute of Technology (Caltech) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Project Description
To be updated.

# Folder Structure
This section will give basic descriptions of the software workflow and what is stored where.
References and adapts [this](https://theaisummer.com/best-practices-deep-learning-code/) article.
Further [best practices](https://neptune.ai/blog/how-to-organize-deep-learning-projects-best-practices).

- configs: in configs we define every single thing that can be configurable and can be changed in the future. Good examples are training hyperparameters, folder paths, the model architecture, metrics, flags.
- evaluation: is a collection of code that aims to evaluate the performance and accuracy of our model.
- executor `[`UNUSED`]`: in this folder, we have the functions and scripts that perform the optimization ~~i.e. train the model or predict something in different environments. And by different environments I mean: executors for GPUs, executors for distributed systems.~~ This package is our connection with the outer world and it’s what our “main.py” will use.
- model `[`UNUSED`]`: ~~: contains the actual deep learning code (we talk about tensorflow, pytorch etc)~~
- trials: contains past trials with all the save data necessary to reconstruct (permittivity data, config etc.)
- utils: utilities functions that are used in more than one places and everything that don’t fall in on the above come here.

![This image](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/DL-project-directory.png) also gives a good overview of what forms folder structure for ~~deep learning~~ optimization code can take.