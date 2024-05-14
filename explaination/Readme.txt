+In equation1 is the future reward in which at time t reward is taken as it is but as we move further in the furure rewards as discounted by gamma to the power of k

+ In equation 2 we have Q function which takes state and action as in put and returns a suggested action, reward is given to a policy denoted by pi using which we define the suggested action (exploration vs exploitation) we decide using epsilon whether to randomly select the action or to take it from the DQN model

+Equation3 simply defines the policy function

+Equation4 denotes the function which returns the expected q value for the given state and action which becomes the label and helps calculate the loss, for the given state S at time t and action 'a' we add reward that we get after performing action 'a' to the discounted (max q value for the future states at different actions) estimated max future q value.
this eqution uses recurssion and estimated value for the future refines as the model trains.

+Equation5 denotes the mean squared error in which we square the difference between label value that was calculated using equation4 and returned q value


watch following playlist to understand in detail
https://www.youtube.com/playlist?list=PL_49VD9KwQ_OpleGtJhWD24JDrrmXYXEQ