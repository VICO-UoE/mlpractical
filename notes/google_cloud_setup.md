# Google Cloud Usage Tutorial
This document has been created to help you setup a google cloud instance to be used for the MLP course using the student credit the course has acquired.
This document is non-exhaustive and many more useful information is available on the [google cloud documentation page](https://cloud.google.com/docs/).
For any question you might have, that is not covered here, a quick google search should get you what you need. Anything in the official google cloud docs should be very helpful.

| WARNING: Read those instructions carefully. You will be given 50$ worth of credits and you will need to manage them properly |
| ---------------------------------------------------------------------------------------------------------------------------- |


### To create your account and start a project funded by the student credit
1. Login with your preferred gmail id to [google cloud console](https://cloud.google.com/), click on Select a Project on the left hand side of the search bar on top of the page and then click on New Project on the right hand side of the Pop-Up.
Name your project sxxxxxxx-MLPractical - replacing the sxxxxxxx with your student number. Make sure you are on this project before following the next steps. 
2. Get your coupon by following the instructions in the coupon retrieval link that you received.
3. Once you receive your coupon, follow the email instructions to add your coupon to your account.
4. Once you have added your coupon, join the [MLPractical GCP Google Group](https://groups.google.com/forum/#!forum/mlpractical_gcp) using the same Google account you used to redeem your coupon. This ensures access to the shared disk images.
5. Make sure that the financial source for your project is the MLPractical credit by clicking the 3 lines icon at the top left corner and then clicking billing -> go to linked billing account.
6. If it's not set to the MLPractical credits then set it by going to billing -> manage billing accounts -> My projects. Click the 3 dots under the Actions column for the relevant project and click change billing account. Select the MLPractical credit from your coupon.
6. Start the project

### To create an instance
1. Click the button with the three lines at the top left corner.
2. Click ```Compute Engine```. You might be asked to activate it.
3. On the left hand side, select ```VM Instances```.
4. Click the ```CREATE INSTANCE``` button at the top of the window. 
5. Name the instance ```mlpractical-1```
6. Select region to be ```us-west1(Oregon)``` and zone to be ```us-west-1b``` (there are other suitable regions however this one has K80s available right now so we went with this one, feel free to find something else if for some reason you need to, but it is recommended ro run on K80 GPUs.)
7. In Machine Configuration, select ```GPU``` machine family.
8. Select NVIDIA Tesla K80. Those are the cheapest one, be careful as others can cost up to 8 times more to run
9. Series and in Machine type select  ```2 vCPUs``` with ```7.5Gb memory```.
10. Under ```Boot disk```, click change.
11. On the new menu that appears, select the ```Deep Learning on Linux``` operating system, with the ```Pytorch 1.10, no-XLA``` version, then click select at the bottom.
12. You should consider going into the ```Networking, disks, security, management, sole tenancy``` drop down menu at the bottom and enable ```Preemptiblity``` under management. As the warning says, this will restrict your VM life to 24h but greatly cuts costs. Using this option will be helpful if you're running low on credits or need quick results in a bind (paired with a stronger GPU)
13. Click ```Create```. Your instance should be ready in a minute or two.
14. If your instance failed to create due to the following error - ```Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally.```,  type ```quota``` in the search bar then click ```All quotas```
15. Search for 'GPUS_ALL_REGIONS' in the filters
16. Tick in the box next to Global and then Click ```Edit Quotas``` in the top bar. 
17. This will open a box in the right side corner asking for your details. Fill in those and then click Next.
18. Put your New Limit as ```1``` and in the description you can mention you need GPU for machine learning coursework. And then Send Request. 
19. You will receive a confirmation email with your Quota Limit increased. This may take some minutes.
20. After the confirmation email, you can recheck the GPU(All Regions) Quota Limit being set to 1. This usually shows up in 10-15 minutes after the confirmation email. 
21. Retry making the VM instance again as before and you should have your instance now. 


#### Note
Be careful to select 1 x K80 GPU (P100s and P4s are 5x more expensive)

You only have $50 dollars worth of credit, which should be about 125 hours of GPU usage on a K80.


### To login into your instance via terminal:
1. In a DICE terminal window ```conda activate mlp```
2. Download the `gcloud` toolkit using ```curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-365.0.0-linux-x86_64.tar.gz```
3. Install the `gcloud` toolkit using ```tar zxvf google-cloud-sdk-365.0.0-linux-x86_64.tar.gz; bash google-cloud-sdk/install.sh```.
**Note**: You will be asked to provide a passphrase to generate your local key, simply use a password of your choice. There might be some Yes/No style questions as well, choose yes, when that happens.

4. Reset your terminal using ```reset; source ~/.bashrc```. Then authorize the current machine to access your nodes run ```gcloud auth login```. This will authenticate your google account login.
3. Follow the prompts to get a token for your current machine.
4. Run ```gcloud config set project PROJECT_ID``` where you replace `PROJECT-ID` with your project ID, you can find that in the projects drop down menu on the top of the Google Compute Engine window; this sets the current project as the active one
5. In your compute engine window, in the line for the instance  that you have started (`mlpractical-1`), click on the downward arrow next to ```ssh```. Choose ```View gcloud command```. Copy the command to your terminal and press enter.
6. Add a password for your ssh-key (and remember it!). 
7. Re-enter password (which will unlock your ssh-key) when prompted.
8. On your first login, you will be asked if you want to install nvidia drivers, agree and make sure the installation runs well.
9. Run ```nvidia-smi``` to confirm that the GPU can be found.  This should report 1 Tesla K80 GPU. if not, the driver might have failed to install. Logout and retry.
10. Well done, you are now in your instance! When you login you may see an error of the form `Unable to set persistence mode for GPU 00000000:00:04.0: Insufficient Permissions` - you should be able to ignore this.  The instance on the first startup should check for the gpu cuda drivers and since they are not there, it will install them.  This will only happen once on your first login. Once the installation is finished you are ready to use the instance for your coursework.
11. Clone a fresh mlpractical repository, and checkout branch `coursework2`: 

```
git clone https://github.com/VICO-UoE/mlpractical.git ~/mlpractical
cd ~/mlpractical
git checkout -b coursework2 origin/mlp2021-22/coursework2
python setup.py develop
```

Then, to test PyTorch running on the GPU, run this script that trains a small convolutional network (7 conv layers + 1 linear layer, 32 filters) on CIFAR100:

```
python pytorch_mlp_framework/train_evaluate_image_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --num_stages 3 --num_blocks_per_stage 0 --experiment_name VGG_08_experiment --use_gpu True --num_classes 100 --block_type 'conv_block' --continue_from_epoch -1
```

You should be able to see an experiment running, using the GPU. It should be doing about 26-30 it/s (iterations per second).  You can stop it when ever you like using `ctrl-c`.

If all the above matches what’s stated then you should be ready to run your coursework jobs.

### Remember to ```stop``` your instance when not using it. You pay for the time you use the machine, not for the computational cycles used.
To stop the instance go to `Compute Engine -> VM instances` on the Google Cloud Platform, slect the instance and click ```Stop```.

#### Future ssh access:
To access the instance in the future simply run the `gcloud` command you copied from the google compute engine instance page.


## Copying data to and from an instance

Please look at the [google docs page on copying data](https://cloud.google.com/filestore/docs/copying-data).

To copy from local machine to a google instance, have a look at this [stackoverflow post](https://stackoverflow.com/questions/27857532/rsync-to-google-compute-engine-instance-from-jenkins).

## Running experiments over ssh:

If ssh fails while running an experiment, then the experiment is normally killed.
To avoid this use the command ```screen```. It creates a process of the current session that keeps running whether
 a user is signed in or not.
 
The basics of using screen is to use ```screen``` to create a new session, then to enter an existing session you use:
```screen -ls```
To get a list of all available sessions. Then once you find the one you want use:
```screen -d -r screen_id``` 
Replacing screen_id with the id of the session you want to enter.

While in a session, you can use 
- ```ctrl+a+esc``` To pause process and be able to scroll
- ```ctrl+a+d``` to detach from session while leaving it running (once you detach you can reattach using ```screen -r```)
- ```ctrl+a+n``` to see the next session.
- ```ctrl+a+c``` to create a new session
 
