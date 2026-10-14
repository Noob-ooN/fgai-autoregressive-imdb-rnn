# Measured experiment report

Selected by lowest validation loss: **LSTM**. Test data was not used for selection.

| Model | Parameters | Best epoch | Train loss | Val loss | Val PPL | Val accuracy | Test PPL | Train s/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Simple RNN | 524320 | 4 | 4.6389 | 4.9772 | 145.07 | 15.46% | 141.73 | 7.39 |
| Bidirectional RNN | 788640 | 2 | 4.9483 | 5.0497 | 155.97 | 14.80% | 151.84 | 12.37 |
| LSTM | 549280 | 4 | 4.7744 | 4.9718 | 144.29 | 15.73% | 142.62 | 10.11 |
| GRU | 540960 | 4 | 4.6362 | 4.9759 | 144.88 | 15.35% | 141.32 | 13.40 |

## Temperature statistics

| Model | Temperature | Mean length | Distinct-1 | Distinct-2 | Repeated bigrams |
|---|---:|---:|---:|---:|---:|
| Simple RNN | 0.2 | 12.000 | 0.792 | 0.868 | 0.132 |
| Simple RNN | 0.7 | 7.667 | 0.954 | 1.000 | 0.000 |
| Simple RNN | 1.2 | 29.222 | 0.926 | 1.000 | 0.000 |
| Bidirectional RNN | 0.2 | 19.889 | 0.568 | 0.727 | 0.273 |
| Bidirectional RNN | 0.7 | 14.556 | 0.807 | 0.972 | 0.028 |
| Bidirectional RNN | 1.2 | 29.000 | 0.914 | 0.997 | 0.003 |
| LSTM | 0.2 | 2.667 | 0.964 | 1.000 | 0.000 |
| LSTM | 0.7 | 12.111 | 0.891 | 0.990 | 0.010 |
| LSTM | 1.2 | 30.222 | 0.919 | 0.997 | 0.003 |
| GRU | 0.2 | 19.667 | 0.621 | 0.740 | 0.260 |
| GRU | 0.7 | 22.778 | 0.837 | 0.994 | 0.006 |
| GRU | 1.2 | 29.667 | 0.918 | 0.997 | 0.003 |

## Generated samples (replicate 0)

### Simple RNN, T=0.2, seed: the movie was

(EOS immediately)

### Simple RNN, T=0.7, seed: the movie was

(EOS immediately)

### Simple RNN, T=1.2, seed: the movie was

(EOS immediately)

### Simple RNN, T=0.2, seed: the story and the characters

are

### Simple RNN, T=0.7, seed: the story and the characters

, the characters , look may see bad and i have been the worst things

### Simple RNN, T=1.2, seed: the story and the characters

, bored better a look may see bad two another combination of the body things causes with tone up sequence . the hunt keeps up drama ! good , the history stone understanding ok standard acting . , this was lost andy , coming off that von it reminds is

### Simple RNN, T=0.2, seed: i really enjoyed this film because

it is a great movie . the main characters are not

### Simple RNN, T=0.7, seed: i really enjoyed this film because

it . so but i have ever seen . the plot is

### Simple RNN, T=1.2, seed: i really enjoyed this film because

it with place about a lot to tell sure seen on no plot up what's the guy with the cop detective of pretty my favorite good awful throughout lacking , team stephen executed

### Bidirectional RNN, T=0.2, seed: the movie was

a great movie . i was a few

### Bidirectional RNN, T=0.7, seed: the movie was

not to watch it , and i up

### Bidirectional RNN, T=1.2, seed: the movie was

bob to my own , night british up

### Bidirectional RNN, T=0.2, seed: the story and the characters

, the acting is the movie . i was watching this movie . the acting was

### Bidirectional RNN, T=0.7, seed: the story and the characters

, the acting is awful . it is a different

### Bidirectional RNN, T=1.2, seed: the story and the characters

, bored better a look may see bad two another two dislike after to things how with tone up visually . the beside keeps up drama ! good , up based stone understanding ok standard acting . , this wives battle andy , but this carol be it him is

### Bidirectional RNN, T=0.2, seed: i really enjoyed this film because

it was a good movie . i was a few . . . . . i was a little movie . i love it

### Bidirectional RNN, T=0.7, seed: i really enjoyed this film because

it . i have watched it all to say that it was

### Bidirectional RNN, T=1.2, seed: i really enjoyed this film because

it also place about a lot to tell sure seen on special or bugs what's up anymore money , versions about of accept my favorite good awful throughout lacking , team stephen executed

### LSTM, T=0.2, seed: the movie was

(EOS immediately)

### LSTM, T=0.7, seed: the movie was

(EOS immediately)

### LSTM, T=1.2, seed: the movie was

bob to open among , with british up

### LSTM, T=0.2, seed: the story and the characters

, the movie is a movie .

### LSTM, T=0.7, seed: the story and the characters

, the big supporting look may see this movie .

### LSTM, T=1.2, seed: the story and the characters

, bored better supporting look may see bad two another york will see to things how with tone up visually . the film's keeps to start ! good , the history stone about ok standard acting . , this was battle andy , is off that von dealing him from

### LSTM, T=0.2, seed: i really enjoyed this film because

it is a

### LSTM, T=0.7, seed: i really enjoyed this film because

it . so but i have ever seen . the plot is

### LSTM, T=1.2, seed: i really enjoyed this film because

it with place about a lot to tell sure seen on they or bugs what's up anymore everything for cop about of pretty my performances good awful throughout lacking , team stephen executed

### GRU, T=0.2, seed: the movie was

a great movie . i was a movie . i was going to

### GRU, T=0.7, seed: the movie was

not great to describe that i don't up

### GRU, T=1.2, seed: the movie was

bob to my own , night british up

### GRU, T=0.2, seed: the story and the characters

are a few , but i don't know it . i

### GRU, T=0.7, seed: the story and the characters

was a big tv version of the original and i still so after it things was with that i did not the big reasons to sit ! it , the movie . i was not many . , this was so well , but i thought it was hooked on

### GRU, T=1.2, seed: the story and the characters

was bored with a look may see bad two another brian dislike the kid things causes with tone up visually . the beside keeps up drama ! good , the history stone familiar ok standard acting . , this wives battle andy , is off that von dealing him ?

### GRU, T=0.2, seed: i really enjoyed this film because

i have seen it . i was a lot of the film . i was a lot of the movie . i was a lot of the movie . i was a movie . i was a lot of the movie .

### GRU, T=0.7, seed: i really enjoyed this film because

it was so about it . it's a wonderful car years .

### GRU, T=1.2, seed: i really enjoyed this film because

it with place about a lot great 2006 radio seen on no or bugs what's up anymore everything years cop about of accept my favorite good awful throughout lacking , team stephen executed because however did it's just live on all is
