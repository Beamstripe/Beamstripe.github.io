---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}



Education
======
* China University of Geosciences (Beijing), Computer Science (currently)

Internship experience
======
* **Turing Sense (Beijing) Technollogy Co.,Ltd.** *2023 Summer*

* R&D Engineer, Visual Algorithm Dept.
  * Utilized Stable Diffusion and ControlNet models to generate images with fixed styles of clothing types, followed by parameter refinement.
	* Fine-tuned AIGC image model parameters to produce stylistically consistent images with varying clothing.
	* Collaborated with business partner Moodytiger to explore the impact of different models and algorithms on generating images with diverse clothing styles.
	* Acquired expertise in image generation, model adjustment, and collaborative research to diversify clothing style images.
	
	

---


* **Hunan Chenfan Soft Technology Information Technology Co., Ltd.** *2023 Spring*
* Development Engineer, Software Development Dept.
  * Conducted an in-depth evaluation of various speech recognition systems, including Baidu’s speech recognition API, to assess their accuracy and efficiency, collaborating with a team to identify optimal solutions.
  *	Played a pivotal role in enhancing the text-to-speech (TTS) voice generation function, working closely with colleagues to clean and label extensive telephone voice samples for the development of an improved operator voice generation model.
  *	Collaborated on a multidisciplinary team to refine and optimize voice generation technology, leveraging data analysis and system evaluation skills.
  *	Acquired valuable skills in speech technology, data management, and model development, fostering a comprehensive understanding of the field.

  
Skills
======
* Programming Skills:
  *	Strong foundation in standard algorithms and data structures, with the ability to tackle programming challenges effectively.
  *	Remarkable expertise in Python and C++ programming, emphasizing object-oriented principles.
  *	Proficient in designing Java-based web and database applications.
  *	Well-versed in front-end development using HTML, CSS, and JavaScript, with a solid grasp of Vue and Spring frameworks.
* Language Proficiency:
  *	IELTS: 6.5  (Listening: 6.5; Reading: 7.5; Writing: 6.5; Speaking: 6.0)


Honors and awards
======
*	Achieved the Third Prize in the National Finals of the Group Programming Ladder Tournament. (2023)
*	Received Honorable Mention Prizes in ICPC Shenyang Station and ICPC Xi’an Station. (2023)
*	Gained a Successful Participant Award in the Mathematical Contest In Modeling. (2022)
*	Secured Second Prize in Class C at the National English Competition for College Students. (2021)
*	Earned the Second Prize in Group B of Science and Technology at the 32nd Beijing Mathematics Competition for College Students. (2021)
*	Captured the Third Prize in non-mathematics at the 13th National Mathematics Competition for College Students. (2021)
*	Won the Third-Class Major-Specific Scholarships. (University-level, 2020 – 2023)
*	Regularly attained top rankings in the China University of Geosciences (Beijing) programming competition. (2020 - 2023)



# Research interests

## Optimal evacuation path planning based on Reinforcement learning

For my school project, team members and I have completed the model building of reinforcement-learning-based path planning. The project mainly used two algorithm, Q-Learning and SARSA algorithm.

### Q-Learning Algorithm

**Architecture**

![image-20231028215122001](https://picbed-1321448974.cos.ap-beijing.myqcloud.com/img/image-20231028215122001.png)

Based on the basic structure of Q-Learning, we devised the architecture for this specific problem. 

**Basic Idea**

In order to evaluate the expectation of the reward, we introduced state-value function $V_\pi(s)$ and action-value function $Q_\pi(s,a)$. Marking $Q^*(s_t,a_t)$ as action-value function taking the optimum strategy, we have reached the following conclusion:
$$
Q^*(s_t,a_t)=Q(s_t,a_t)+\alpha[r_t+\gamma\max_{a_{t+1} }(Q(s_{t+1},a_{t+1}))-Q(s_t,a_t)]
$$

### SARSA Algorithm

**Basic Idea**

The idea behind SARSA is similar to Q-Learning, and the difference between them is how to actually choose strategy in practice. Instead of scanning through all states and find the best one to proceed in Q-Learning, SARSA opts for a more mild and robust approach. SARSA goes along with a optimal choice first and update Q function after that. Moreover, as we have encountered issues of low convergence rate, we further introduced $\varepsilon$-greedy strategy to update Q functions with a probability, avoiding trapping in the local optimum.

![image-20231028220455863](https://picbed-1321448974.cos.ap-beijing.myqcloud.com/img/image-20231028220455863.png)

### Demonstration

We have experimented and compared the correctness and efficiency of the two algorithm. Using Gym, we have built a sample scenario for the problem. The result is as follows: (Using Q-learning as example)

![demo](https://picbed-1321448974.cos.ap-beijing.myqcloud.com/img/demo.gif)

This is the training result after around 50 epochs.

### Image Data Augmentation Algorithm

Currently, I am actively involved in researching image augmentation algorithms. Collaborating with my esteemed tutor [Liupin][liupin@cugb.edu.cn], we are exploring a novel approach to image mixing for data augmentation, building upon the established SnapMix and other mix methods.

