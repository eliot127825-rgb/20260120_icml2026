## 🎥 Overview Demo

<p align="center">
  <img src="./assets/Demo_Video.gif?raw=1" width="80%">
</p>

<p align="center">
  <sub>📁 Path: <code>./assets/Demo_Video.gif</code></sub>
</p>

## 🎬 Demo

<p align="center">
  <img src="./assets/demo2.gif?raw=1" width="60%">
</p>

<p align="center">
  <sub>📁 Path: <code>./assets/demo2.gif</code></sub>
</p>

### 📝 Model Output

**Question:** Is the man on the left surprised when the man on the right suddenly moves towards him with the pie?

**Options:**
- A. No, he remains calm.
- B. The man on the left is surprised but not scared.
- C. The man on the left is surprised but amused.
- D. No, he looks surprised.

---

**🎯 Context**

> The man on the left is wearing a plaid shirt and a striped tie, while the man on the right is in a dark shirt and a yellow striped tie, holding a pie. The man on the left shows no signs of fear or surprise, but his expression changes to surprise and he looks confused. The man on the right suddenly moves towards the man on the left with the pie, creating a sense of suddenness in the scene.
>
> The man on the left speaks in a hurried and excited tone, emphasizing the importance of the topic. The man on the right responds calmly and gently, slowing his speech to ease tension. The conversation involves a debate about the choice between cake and pie, with the man on the right eventually winning the argument and using the pie as a prop to make a point.

**💭 Thinking**

> As I watch the video, I notice the man on the left in a plaid shirt and striped tie. His expression is calm, and he seems focused on the conversation. However, when the man on the right, who is holding a pie, suddenly moves towards him, his expression shifts. I see him go from calm to surprised, and he looks a bit confused. This sudden action definitely caught him off guard.
>
> Now, listening to the audio, the man on the left is speaking in a hurried and excited tone, which suggests he's really into the topic. The man on the right, in the dark shirt and yellow striped tie, responds calmly and gently, even slowing his speech down, which seems to ease any tension. They're talking about a debate between cake and pie, and the man on the right eventually wins and uses the pie to make a point.
>
> So, considering his calm demeanor before the sudden action and his surprised, confused look afterward, I conclude he was definitely taken aback by the sudden movement with the pie, but he didn't show fear. He was surprised, but not scared.

**✅ Answer: A**

---

**📊 Summary & SAM3 Segmentation**

> **Key Points Analysis**
> 1. The man in the plaid shirt appears calm and neutral, with no signs of surprise.
> 2. The man in the dark shirt suddenly moves with the pie, causing the other man to react.
> 3. The conversation does not suggest any anticipation for the pie incident.
>
> **Video Focus Objects**
> - `person`: man in plaid shirt, with a calm and neutral expression
> - `person`: man in dark shirt, moving suddenly with the pie
>
> **Emotional Indicators**
> - The man in the dark shirt shows sudden movement and possibly surprise
> - The man in the plaid shirt remains calm and composed
>
> **SAM3 Segmentation Targets**
> ```
> person: man in plaid shirt
> person: man in dark shirt, moving suddenly with the pie
> ```

---

### 🎯 Segmentation Comparison

<table>
  <tr>
    <td align="center"><b>Original Video</b></td>
    <td align="center"><b>SAM3 Segmentation</b></td>
  </tr>
  <tr>
    <td><img src="./assets/demo2.gif?raw=1" width="100%"></td>
    <td><img src="./assets/demo2_seg.gif?raw=1" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><sub>📁 <code>./assets/demo2.gif</code></sub></td>
    <td align="center"><sub>📁 <code>./assets/demo2_seg.gif</code></sub></td>
  </tr>
</table>