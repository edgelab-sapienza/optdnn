## TF Optimizer

  

This service provides API endpoints that can be used to optimize models.

  

## Endpoints:

-  **add_task [POST]**

    Add a task in the processing queue, task information have to be passed with a JSON input, the fields are:

    -  **model_url***
       - String with the URL used to download the .keras model
       -  `http://modelhost.com/mymodel.keras`

    -  **dataset_url***
       - String with the URL used to download the .zip file of the dataset
       -  `http://datasethost.com/myhost.zip`
       - [Addition details on *Dataset file structure* section]

    -  **dataset_scale***
       - Pair of ints that represent the range of the expected values by the model
       -  `[-1, 1]`
    -  **img_size**
       - Pair of integers that represent the image size expected by the model (default auto-detected)
       -  `[224, 224]`
    -  **remote_nodes**
       - List of IP addresses and port of the remote nodes used to test the model
       -  `[["192.168.178.2", 12345], ["192.168.178.65", 12345], ["192.168.178.96", 12345]]`
    -  **callback_url**
       - String representing the URL called when the optimization is completed
       -  `http://192.168.178.3:8080/callback?id=3`
       - In the request a parameters with value `download_url` is added, it contains the URL necessary to download the optimized model. This URL can be get also in the information retrieved by the requests `get_tasks` and `{task_id}/info`
    -  **batch_size**
       - Integer representing the batch size, default is 32
       -  `32`

  

(*) required fields

-  **get_tasks [GET]**
    
    Get information of all the task in the queue

-  **{task_id}/info [GET]**

    Get information of a single task

-  **{task_id}/delete [GET]**

    Delete a task

-  **{task_id}/resume [GET]**

    Resume a task

-  **{task_id}/stop [GET]**

    Stop a task

-  **{task_id}/download[GET]**

Delete the optimized model of a task
  

## Dataset file structure

Files inside the dataset zip must be organized in this way:
- Zip package [containing all the folders of the different clusters]
  - 0
    - img0.jpg
    - img1.png
  - ...
  - 1
  - ...
  - 1000

Images can be both .png and .jpg with different sizes and range values, they will be resized by the tool
