---
title: "Learning Markdown Language"
author: "Prince John"
date: 2025-08-11
tags: ["markdown", "documentation", "webdev"]
---
This is the start of the document content.

---

---

# Table of Content h1
## Table of Content h2
### Table of Content h3
#### Table of Content h4
##### Table of Content h5
###### Table of Content h6

[toc]
# Introduction
ivde thodakkam varanam
# Problem
ivde preshnam varanam
# A Solution
ivde solution varanam
# Conclusion
ivde conclusion varanam

---

| Left-Aligned Header | Center-Aligned Header | Right-Aligned Header |
| :------------------ | :-------------------: | -------------------: |
| Content 1           | Content 2             | Content 3            |
| Content 4           | Content 5             | Content 6            |

---

This is a ~~mistaken~~ word.

---

Here's a sentence Term 1 with a footnote.[^1]
Here's another one.[^2]

---

I get a lot of information from [Google][1] and [Wikipedia][2].

---

# My Interactive Document

This is a regular paragraph in Markdown.

<MyInteractiveChart data={myData} />

Here's another paragraph. You can mix and match content easily.

---

X^2^a
X~2

(a + b)^2^ = a^2^ + 2ab + b^2^
H<sub>2</sub>O

---

$$
\frac{-b  \pm \sqrt{b^2 - 4ac}}{2a}
$$

---

```mermaid
graph TD
    A[Start] --> B{Is it a question?};
    B -->|Yes| C[Answer the question];
    B -->|No| D[Move on];
```

---

<details>
  <summary>Click to expand</summary>
    This is the hidden content. It can contain any Markdown-formatted text.
</details>

---

> [!NOTE]
> This is an important note.

> [!WARNING]
> This is a warning. Warnings are for important information that could lead to issues.

> [!QUOTE]
> “The only way to do great work is to love what you do.”
> — Steve Jobs

---

I'm so happy! 😄

---

Press `<kbd>`Ctrl `</kbd>` + `<kbd>`C `</kbd>` to copy.

---

Term 1
: Definition of the first term.

Term 2
: Definition of the second term, which can span multiple paragraphs.

---

// Fenced code block with line numbers
// The syntax for this varies, but it's often a special keyword.
// e.g.,

```js
function greet(name) {
  console.log(`Hello, ${name}!`);
}
```
---

[Go to Introduction](#introduction)

---

[1]: https://www.google.com
[2]: https://www.wikipedia.org
[^1]: This is the first footnote.
    
[^2]: This is the second footnote.


