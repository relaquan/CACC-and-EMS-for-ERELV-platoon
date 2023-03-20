# Collaborative adaptive cruise control and energy management strategy for extended-range electric logistics van platoon

This paper improves the economy of the extended-range electric logistics van (ERELV) platoon from two aspects of cooperative adaptive cruise control (CACC) and energy management strategy (EMS).
Based on the DMPC algorithm, an Eco-CACC controller with the optimization goals of stability, comfort and energy-saving is designed.
The multi-agent deep reinforcement learning (MADRL) algorithm is used to solve the EMS of the ERELV platoon.

## Eco-CACC and MADRL EMS overall control framework
![论文图片](https://user-images.githubusercontent.com/69177652/225628593-3d345c6e-bc35-4cf5-81fa-339341a6799a.png)
 
## Eco-CACC based on DMPC 
A single-point optimization problem is defined on each vehicle node in the platoon, and the model prediction optimization problem of all nodes needs to be solved at each optimization moment.  Using the domain node information obtained by the PLF communication topology, each sub-problem is optimized to obtain the control input of the vehicle.
![image](https://user-images.githubusercontent.com/69177652/226297689-92c32791-68aa-42fa-b71d-669640bc879e.png)

## Eco-CACC based on DMPC 
