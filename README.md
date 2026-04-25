# chatGnT

chatG&T - an autoregressive GPT model for cocktail recipe generation.

Course project for spring 2026 CSCI 5802 data science course at CU Boulder.

## **Installation**

```bash
git clone git@github.com:sdslack/chatGnT.git
cd chatGnT
pip install .

```

## **User Guide**

ChatG&T is a command-line tool that generates cocktail recipes based on user
input.

Give it your the name of your favorite cocktail to generate a new recipe that's
 similar to what you already like:

```bash
chatgnt --input "margarita"

```

```text
['1.5 parts tequila', '0.5 part triple sec', '0.75 part sweet vermouth',
'1 dash angostura bitters']

```

You can run this repeatedly to see different recipes!

Instead of a cocktail name, you can also input an ingredient to generate new
recipes that include it:

```bash
chatgnt --input "gin"

```

```text
['1 part gin', '0.75 part sweet vermouth', '0.5 part triple sec',
'0.5 part lemon juice']

```

Finally, you can specify which chatG&T model to use - single-task (st) or
multi-task (mt). The single-task model is trained using standard autoregressive
next-token prediction with forced alternating structure output, while the
multi-task model is trained with two output heads to generate an amount and
an ingredient at each model step. By default, the chatG&T uses single-task.

```bash
chatgnt --input "gin" --model-type "mt"

```

```text
['1 part gin', '1 part light rum', '1 part triple sec', '0.25 part sour mix']

```

## **Future TODOs**

+ Cache loaded data/models during CLI usage, remove Kaggle dependence
+ Add more detailed error handling when drinks/ingredients not found
+ If develop/polish further, add to PyPI for easier installation
