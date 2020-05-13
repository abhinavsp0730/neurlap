---
toc: true
comments: true
image: images/article5.jpeg
layout: post
description: Easily understand the backpropagation algorithm in a way you’ve never before!

title: Understanding the Backpropagation Algorithm
---
### Intro to Backpropagation 101 | [Towards AI](https://towardsai.net)

## Understanding the Backpropagation Algorithm

### ***Easily understand the backpropagation algorithm in a way you’ve never before!***

You might have been taking a course on deep learning and at the beginning of it, it seems to you very easy, and then you have encountered “backpropagation” and you will start your head scratching because it is too “*mathsy*.”

![](https://cdn-images-1.medium.com/max/2000/1*wr-K2ZlmXY-5M9y1J-OUXg.gif)

**“why there is a need for understanding the Back Propagation algorithm for us?**
The answer is very simple. We, human beings are always curious to know how things are happening in the real world.
 For example, when we see human breathing. From outside it looks like it is simply the intake of air & release of CO2.
 
But, the curiosity of the human race led us, to know that, in this short interval of 1–2 sec.
The whole blood id first oxygenated & then it transported through nerves and reaches to every single cell of our body.
And finally, it deoxygenated.

Humans are always curious to know, how the mechanism which governs the things of the real world around us.
 And for any Deep Learning practitioners our,” Our world revolves around neural networks”.
So, for the sake of killing your curiosity & and giving you the idea of how neural networks are trained.
 I’m presenting you with this article.

**The context of this article are:-**
1. Giving you a brief intro about the neural networks.
2. Try to make you understand Back Propagation in a simpler way.
3. And, finally, we’ll deal with the algorithm of Back Propagation with a concrete example.

Okay! So, first understand what is a neural network.

I don’t know you are aware of a neural network or not. So let us first understand this concept.

![](https://cdn-images-1.medium.com/max/2000/1*rjhM89acID1wzciUFNjyhA.gif)

*A neural network is a series of algorithms that endeavors to recognize,
underlying relationships in a set of data through a process that mimics the way the human brain operates.
Neural networks can adapt to changing input; so the network generates the best possible result without needing to redesign the output criteria.*

or in a simple words, we can describe a neural networks as:-

-It’s just a computer program that learn and behave in a remarkably similar way to human brains.

- Or, it’s just a way how computers learn things, recognize patterns, and make decisions in a human like way.

- Or, neural networks enable computers to learn from a given sent of data.

-Or, ( in a more elaborative way ) An ANN is simulation of the network of neurons that make up a human brain so that the computer will be able to learn things and make decisions in a human like manner.

*In simpler words, Back Propagation is the central mechanism to train a neural network. In which we calculate the error of our desired targeted output value & then we adjust the weights in a way to minimize this error.
it’s similar to human being how we learn from our mistakes.*

Try to imagine the situation. You are trying to hit a football into the goal post.
you randomly kick the football with some angle. Then you measure the distance from the goalpost to the spot where your football goes
in the first attempt.
 Then we try to minimize this distance by changing the angle by which we kicked the football.
And finally, we’re able to hit the football into the goal post.

Now, compare this situation with a neural network:-
1. we randomly kicked the football.
 In a neural network, we initialized the weights randomly.

2. we measure the distance from the goal post & the spot where we hit the ball.’
 In a neural network, this is called measuring the total error.

3. We changed the angle in such a manner to minimize this distance

In a neural network, this is called updating the weights in order to get the desired targeted output.

The Backpropagation algorithm is a very powerful algorithm in order to train a neural network.
it’s so powerful that it is used in Zip Code recognition(low-level example),Face recognition (mid-level example) to Sonar target recognition(high-level example).

I hope now you’ve understood Back Propagation.
So, now you are ready to deal with the mathematical stuffs of Back Propagation.

### Okay so, now jump into Backpropagation algorithm to understand it.

![](https://cdn-images-1.medium.com/max/8320/1*vN-q9t2GAQb1ZH2O_VUQ7A.jpeg)

This is a figure of a simple neural network having 2 layers i.e input, hidden and output layer, respectively. Each layer is having 2 neurons.

![](https://cdn-images-1.medium.com/max/2160/1*ye2axME8P0tP3q1TkzVi5w.png)

The function of a neuron is, to sum up, all the multiplied inputs with its weight & the bias.
And the Output is followed by the operation of the activation function.

### Let us understand Back Propagation with an example:

![](https://cdn-images-1.medium.com/max/8320/1*aS1zaM91wXrwwOERRtO6FQ.jpeg)

![](https://cdn-images-1.medium.com/max/8320/1*TUSSNWZfo_S3aSUwzorT3A.jpeg)

*Here,H1 is a neuron and the sample inputs are x1=0.05,x2=0.10 and the biases are b1=0.35 & b2=0.60.
 The targeted values are T1=0.01 & T2=0.22*

### Now we randomly initialize the weights,

![](https://cdn-images-1.medium.com/max/8320/1*QybSyhaYUA_2f2WcFCAAPA.jpeg)

**Note:** *In this whole article we’re using SIGMOID as an activation function.*

![](https://cdn-images-1.medium.com/max/8320/1*28fuQfhpPTq0iYSx9I2kBg.jpeg)

### Let us calculate H1, H2 and output H1, output H2.

![](https://cdn-images-1.medium.com/max/8320/1*xEMJqOHzb2vdk9ZAGj6wEA.jpeg)

### Similarly, we can calculate y1,y2, output y1 & output y2.

![](https://cdn-images-1.medium.com/max/8320/1*5R4N2g9XJRXDXNnmZ5PnJg.jpeg)

### Calculating the total error

![](https://cdn-images-1.medium.com/max/8320/1*wXYkRb-Bgyz1W-eUAO1dOQ.jpeg)

### Now we have to backpropagate, to upgrade the weights

*Consider w5, Error at w5*

![](https://cdn-images-1.medium.com/max/8320/1*ouMKSO9lKY_DFVCgPp42Aw.jpeg)

*But there is no term w5 present in the expression of Etotal.
So we have to split it & apply the chain rule to partially differentiate it.*

### Splitting

![](https://cdn-images-1.medium.com/max/8320/1*eowi-PL_abyOD6jd0jxavw.jpeg)

*Partially differentiating each term one by one.*

![](https://cdn-images-1.medium.com/max/8320/1*vnhrT3bVfmaQCGh24W1Sig.jpeg)

![](https://cdn-images-1.medium.com/max/8320/1*wC23x9yYyAxyXbiEChDdiw.jpeg)

![](https://cdn-images-1.medium.com/max/8320/1*K67OxPb2B_hzoLVSl6Vc-g.jpeg)

### Calculated error w5:-

![](https://cdn-images-1.medium.com/max/8320/1*G9twD4Sz1Us-tA9mCWmE1g.jpeg)

### Now updating w5

![](https://cdn-images-1.medium.com/max/8320/1*OyOKpPKcLYTk6Xg1rTHZeg.jpeg)

*New updated weights,w5=0.3595 & similarly w6=0.4086,w7=0.511 & w8=0.561 .*

### Now at hidden layer,updating w1,w2,w3&w4.
Consider w1
Error at w1

![](https://cdn-images-1.medium.com/max/8320/1*bwSkuaUXJjEfmMjQwX92xg.jpeg)

*But there is no w1 term present in the expression of Etotal.
 So, in order to do that, we have multiple splits.*

## Please pay attention and look it slowly!

*The terms which are encircled, we can’t differentiate them directly. So, we have to split them.*

![see the terms which are encircled.](https://cdn-images-1.medium.com/max/8320/1*bwSkuaUXJjEfmMjQwX92xg.jpeg)

### **Consider the term which is encircled orange & let us split & apply the chain rule.**

![](https://cdn-images-1.medium.com/max/8320/1*SjJR9eI_47zdG4U0-poKTw.jpeg)

### Now calculating:

![](https://cdn-images-1.medium.com/max/8320/1*aVxJV0pisKrBtg29uGliYQ.jpeg)

![](https://cdn-images-1.medium.com/max/8320/1*SggwwZ6oa1EigDTkTC8f7Q.jpeg)

### We have calculated the term which is encircled orange.

![](https://cdn-images-1.medium.com/max/8320/1*y1UU4A0ohRRCyskoleB2GA.jpeg)

### Consider the term which is encircled in blue.

![](https://cdn-images-1.medium.com/max/8320/1*oU08xnO9QU7f9Vf02z3JxA.jpeg)

### Consider the term which is encircled in black.

![](https://cdn-images-1.medium.com/max/8320/1*zPZ6GktQB7TRKRDUJq-Npg.jpeg)

### Now we’ve calculated all the terms which are encircled.
 Therefore the error at w1 is:

![](https://cdn-images-1.medium.com/max/8320/1*vMUXSRVpon9rKpYPVsGqcQ.jpeg)

### Updating w1:

![](https://cdn-images-1.medium.com/max/8320/1*8VNmSD4aNRTHp_-fV1KPKA.jpeg)

*Now, we are having our updated weight w1 and similarly, we can calculate w2,w3 &w4. What we’ve done so far is we Back-Propagated and updated first w5,w6,w7,w8 and then with the help these we further Back-Propagated and updated the weights w1,w2,w3 & w4.*

So with these updated weights(w1,w2,w3 & w4).We’ve to again calculate H1 & H2.After calculating H1 & H2 ,we can calculate y1 output & y2 output. After that, we can calculate the Total Error as we have done earlier, and again with the help of this new Total Error. We backpropagate & updated the weights.

![](https://cdn-images-1.medium.com/max/8320/1*wAzdjq5phU6a1NmUugxR7A.jpeg)

### And again with the updated weights we forward propagate.

![](https://cdn-images-1.medium.com/max/8320/1*XGW_s1iM_rdDiHDmNdyYyw.jpeg)

*We’ve to iterate over & over again between Back Propagation & forward propagation.*

![](https://cdn-images-1.medium.com/max/8320/1*wAzdjq5phU6a1NmUugxR7A.jpeg)

![](https://cdn-images-1.medium.com/max/2000/1*6z6cpjseugYHYHFZQcOnkQ.gif)

![](https://cdn-images-1.medium.com/max/8320/1*XGW_s1iM_rdDiHDmNdyYyw.jpeg)

*Until the Total Error(cost function) is minimized or in other words, the value of our predicted outputs is closer to that of the target values.*

In one sentence we can define backpropagation **as *it’s a common method of training a neural net in which the initial system output is compared to the desired output, and the system is adjusted until the difference between the two is minimized.*
> # We can now say that backpropagation is the central mechanism to train any neural network.

I hope you’ve understood now what is the backpropagation algorithm and how it works.

Congratulations you’ve just understood one of the toughest “*mathsy”* topics of machine learning.

Don’t forget to give us your 👏 & follow me!!!!!!!

![please clap and follow me!!!!](https://cdn-images-1.medium.com/max/2000/1*HnhqbqJ1vlHFEZmO5oEtqQ.gif)
