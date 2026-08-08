# The simple method: put both on a millimetre grid, then slide one over the other

*A guide to `registration_idea.py`. Written to be read with no prior background.
Companion to `registration_gates_docs.md`, which explains what went wrong with
the complicated method and is worth reading first if you have not.*

---

## 1. The problem, in one paragraph

You have two pictures of the same person: a CT and an MRI, taken on different
machines on different days. You want to know which pixel in one corresponds to
which pixel in the other. They are not the same size, they were not taken from
the same position, and the patient was lying slightly differently. Something has
to work out how to move one picture so it sits correctly on the other.

## 2. Why we are trying something simpler

The method used until now can slide, turn, resize and skew the MRI. Six numbers
worth of freedom. When it was run over 33 slices, **24 of them came back with
something wrong**: pictures turned 21 degrees, one smeared into diagonal stripes,
several with most of the image pushed off the edge of the frame. In every case
the score said things had got *better*.

The full account is in `registration_gates_docs.md`. The short version is that
giving the machine six ways to move a picture gives it six ways to cheat, and
the score cannot tell the difference between "lined up properly" and "damaged in
a way that happens to score well".

So: what happens if we do not give it those six ways? What is the least we can
get away with?

---

## 3. The whole method

Two steps.

> **Step 1.** Resample the CT and the MRI so that one pixel means exactly one
> millimetre in both.
>
> **Step 2.** Slide the MRI over the CT one whole pixel at a time, try every
> position, and keep the one that scores best.

That is genuinely all of it. No starting guess. No repeated attempts from
different places. No coarse-then-fine. No rules about which answers are allowed,
because there are no bad answers available.

---

## 4. Step 1 — what "one pixel = one millimetre" means

### A picture does not know how big it is

Open a photograph and you have a grid of coloured squares. Nothing in that grid
tells you whether it shows a coin or a car park. Size is not in the picture.

A medical scan carries the missing information separately, in a field called
**pixel spacing**: "each of my pixels is 0.488 mm across". A different scan says
"each of mine is 0.875 mm across".

### Why that ruins a comparison

Take a structure 50 mm wide — a bone, say.

- In the CT at 0.488 mm per pixel, it covers about **102 pixels**.
- In the MRI at 0.875 mm per pixel, it covers about **57 pixels**.

The same bone. Two very different pixel counts. If you lay one picture on the
other and compare pixel to pixel, you are comparing a stretched version of the
bone to a squashed one, and no amount of sliding will make them agree.

**This is the single thing that makes registration hard, and it is fixable
before you start.**

### The fix

Resample each picture so one pixel becomes exactly one millimetre. The CT shrinks
from 512 pixels to 250. The MRI changes from its own size to its own millimetre
size. Now:

```
                     before                          after
  CT    512 x 512 px at 0.488 mm     ->     250 x 250 px at 1 mm
  MRI   208 x 320 px at 0.875 mm     ->     182 x 280 px at 1 mm
```

**Now the 50 mm bone covers 50 pixels in the CT and 50 pixels in the MRI.**

Two things follow from that, and both are worth stating plainly:

1. **There is no size difference left.** Nothing to correct. So we do not need
   the machinery that corrects it, and we do not need the rules that stop that
   machinery misbehaving.
2. **The pictures are now in real units.** "Move it 23 pixels" and "move it
   23 millimetres" become the same sentence. You can check the answer against
   what the scanner recorded about where the patient was.

Notice that the two pictures end up *different shapes* — 250×250 and 182×280.
That is correct and expected. They cover different amounts of the body. They are
simply now measured in the same units, which is the only thing we needed.

---

## 5. Step 2 — sliding

### What a whole-pixel slide is

Lay the MRI on top of the CT. Nudge it one pixel left. Score it. Nudge it one
pixel further. Score it. And so on, over a square of possible positions.

Because a pixel is a millimetre, "13 pixels down and 4 across" *is* "13 mm down
and 4 mm across".

### Try all of them — coarsely first, then finely

The obvious thing is to try **every** position. At the default range of 90 mm
each way that is 181 × 181 = **32,761 positions**, which works but takes about
two minutes a slice.

The program now does it in two passes:

1. **Sweep the whole square in steps of 4 mm.** About 2,200 positions.
2. **Take the best 5 of those, and search every single pixel around each one**
   — a 9 × 9 box, which is exactly the gap the 4 mm steps left unexamined.

Around **2,400 positions instead of 32,761 — 13.6 times less work.**

Why the best *five* and not just the best one? Because the coarse sweep only
samples the landscape. The position that came second on a rough look can be the
true winner once you look closely. Refining five costs about 200 extra
positions, which is nothing, and it means a near-miss is not thrown away.

This buys most of what a full search buys:

**There is no starting guess.** The usual approach begins somewhere and walks
uphill, and where it starts changes where it ends up. Here there is no start —
the coarse pass looks everywhere.

**It gives the same answer every time.** No random numbers anywhere. Run it
twice, get the same result. If it is wrong, it is wrong the same way every time,
which is the kind of wrong you can investigate.

**What it gives up, stated plainly.** A full search could promise "this is the
best position that exists". Stepping in 4 mm jumps cannot quite promise that,
because a peak narrower than 4 mm could be stepped over. Two things make that
unlikely: anatomy at 1 mm is many pixels across, so the score changes smoothly
rather than in one-pixel spikes; and refining the best five catches a peak that
merely looked unpromising at first.

**And it was checked rather than assumed.** Both methods were run over all 33
slices:

```
identical shift found : 33 / 33
identical score       : 33 / 33   (to nine decimal places)
positions evaluated   : 1,081,113  ->  79,533
```

Set `COARSE = 1` to turn the shortcut off and reproduce the full search.

---

## 6. What this method cannot do

It can only slide. It cannot turn, resize or skew the picture.

**So if the patient was genuinely lying at a slight angle, this will not fix it.**
It will find the best available slide and stop, and a few degrees of tilt will
remain.

That is the trade, stated honestly:

| | complicated method | this method |
|---|---|---|
| corrects sliding | yes | yes |
| corrects turning | yes | **no** |
| corrects size | yes | not needed — step 1 handled it |
| corrects skew | yes | **no** |
| can produce a ruined picture | **yes — did so, 24 times in 33** | no |
| same answer every run | no | yes |

The last two rows are why this is worth considering. **A method that cannot
express a bad answer does not need to be watched.** You do not need a rule
against turning too far when turning is not available.

---

## 7. Two details that do more work than they look like

These are the parts that would be easy to get wrong, and getting them wrong
would quietly reintroduce the problem we are trying to escape.

### 7.1 Score the same pixels every time

The score is worked out by comparing pixels. If you let the *set* of compared
pixels change as the picture moves, you have a problem — and it is a subtle one.

Think of it as an exam. The score is the percentage you got right. Now imagine
you are allowed to cross out any questions you got wrong before handing the paper
in. Your percentage goes up every time. You have learned nothing.

That is exactly what happens if pixels stop counting when the MRI moves off the
edge of the frame. And it is worse than random, because **the pixels that leave
first are the corners and edges — the ones least likely to have a matching
structure, which is to say the ones dragging the score down.** Throwing away
your worst answers raises your average.

This was measured on the complicated method. Across 33 slices, the connection
between "how much of the picture was thrown away" and "how much the score rose"
was **+0.51**. The more a transform destroyed, the better it scored.

So here, the pixels compared are **the whole CT frame, every time, for every
candidate position**. That set never changes, so nothing can be crossed out.

> **A mistake worth learning from.** The first version of this file did something
> extra: it also trimmed a margin off the CT, the size of the search range, "to
> be safe". That was pointless. The pixel set was already fixed — it does not
> depend on where the MRI is. All the trimming achieved was scoring **31% of the
> CT on average, and 11% on the worst slice**, and producing pictures with the
> edges cut off. A safety measure that protects nothing still costs something.
> It was removed.

### 7.2 Give "no MRI here" a category of its own

When the MRI is shifted, part of the frame has no MRI under it. Those pixels
still have to be dealt with, and there are two choices:

- **Skip them.** This is the crossing-out problem again, through a side door.
- **Count them as their own kind of value.** "There is nothing here" becomes a
  real answer, alongside "this is bright" and "this is dark".

We do the second. It costs one extra row and column in the counting table. The
effect is that a position which pushes half the MRI off the frame is *charged*
for it, in a way the score can see.

### 7.3 So coverage does drop, and that is fine

Because of this, one number in the output changes as the shift grows:
**coverage**, meaning how much real MRI is actually under the frame. Slide a
picture 60 mm and part of it leaves. Coverage falls. That is arithmetic, not a
fault.

The question is not *does coverage change* — it must. The question is **does
losing coverage pay?**

Giving it its own category charges a position for what it moves out of view, so
the incentive is weakened. **It is not removed** — see section 11, where the
measured numbers are set out and an earlier, stronger claim made here is
withdrawn. What the method really provides is a limit on how much can be lost at
all: at worst about a tenth of the picture, against nineteen-twentieths for the
complicated method.

That is the number to re-check if you change anything in this file.

---

## 8. The one thing that can go wrong

Only one, and it is the good kind: **it tells you when it happens.**

If the true offset between the two scans is larger than the search square, the
best position will sit on the **edge** of the square — not because it is a good
position, but because the program was not allowed to look any further. The
answer is a wall, not a hilltop.

So every result records `hit_edge`. When it is true, the picture gets a red
warning printed on it and the fix is one number:

```python
SEARCH = 90     # millimetres each way
```

This is worth appreciating for what it is. The failures in the complicated
method were silent — a 21-degree turn and a smeared picture both came back
labelled "improved". **This failure raises its hand.** A method that tells you
when it is out of its depth is worth more than one that quietly guesses.

---

## 9. What it costs

A full search grows with the **square** of the range. The two-pass search grows
much more slowly, because only the coarse sweep covers the whole square and it
does so in steps:

| range | full search | two-pass (step 4, best 5) |
|---|---|---|
| ±40 mm | 6,561 | ~800 |
| ±60 mm | 14,641 | ~1,300 |
| ±90 mm | 32,761 | **~2,400** |
| ±180 mm | 130,321 | ~8,500 |

Measured on this dataset, the two-pass search took **8–18 seconds a slice**
against 103–265 seconds for the full one.

Doubling the range still quadruples the coarse sweep, so it is worth setting the
range from what `hit_edge` reports rather than picking a large number in case.
But the penalty for being generous is now much smaller than it was.

---

## 10. How to read the output

```
  CT   512x512 px at 0.488 mm  = 250x250 mm
  MRI  208x320 px at 0.875 mm  = 182x280 mm

  after resampling to 1 mm/px:  CT 250x250   MRI 182x280   (1 px = 1 mm in both)

  scored window          = 250x250 px, identical for every shift
  shifts evaluated       = 32761  (+/-90 mm on each axis)
  baseline NMI (no shift)= 1.060968
  best NMI               = 1.115285
  shift found            = -1 mm across, -23 mm down
  change                 = +0.054318
```

| line | what to make of it |
|---|---|
| the two size lines | check these first — if a picture's millimetre size looks wrong, the pixel spacing in the file is wrong, and nothing after this matters |
| `scored window` | should be the full CT, and identical for every shift |
| `shift found` | a physical statement. Compare it against what the scanner recorded about patient position |
| `change` | how much better than not moving at all. Should be positive; it cannot be negative, since "do not move" is one of the positions tried |

### The pictures

Four panels per slice, in `registration_demo_output/sweep_idea/`:

**CT** · **MRI after shifting** · **overlay before** · **overlay after**

In the overlays the CT is amber and the MRI is cyan. Where they agree, the edges
sit on top of each other and the result looks pale and neutral. Where they
disagree you get a coloured fringe — amber on one side, cyan on the other.

**How to read a fringe.** An even halo all the way round means a small leftover
offset. A fringe that is thick on one side and absent on the opposite side means
the two pictures are at an **angle** to each other — and that is the one thing
this method cannot fix. If you see that, you have found a slice that genuinely
needs turning, and that is useful information rather than a failure.

---

## 11. Results over the 33 slices

All 11 series, first/middle/last slice, ±90 mm search, whole CT frame scored.

```
delta NMI               : mean +0.0262   max +0.0750   min +0.0028
improved / unchanged    : 26 / 7        (unchanged = gain below MIN_GAIN of 0.010)
worse than doing nothing: 0
shift found             : mean 20.6 mm   max 76.7 mm
best shift on boundary  : 0 / 33
coverage lost to shift  : mean 0.98%     max 10.42%
```

**Nothing went backwards.** That is not luck — "do not move" is one of the
positions tried, so the answer can never be worse than the starting point. It is
a property of searching everything rather than a result.

**Nothing hit the search boundary.** ±90 mm was enough for every slice, and the
largest genuine offset found was 76.7 mm. At ±60 mm, three slices had come back
sitting on the wall; raising the range fixed all three.

**The gains are smaller than the complicated method's** — mean +0.026 against
+0.067. That comparison flatters the other method, because a large part of its
average came from slices where it was raising its score by damaging the picture.
A smaller honest number is worth more than a larger dishonest one.

### The crop was changing the answers, not just the pictures

Before the trimming bug was fixed, the score only saw the middle 31% of the CT
(11% on the worst slice). It did not merely produce cropped pictures — **it
produced different answers**, because aligning on the middle of a frame is not
the same as aligning on the anatomy:

```
knee / coronal / middle
  scoring 111x111 px (cropped)   ->   shift (-57, -44) mm
  scoring 231x231 px (full)      ->   shift (-10, +13) mm
```

Those are 70 mm apart. Worth remembering the next time a change looks purely
cosmetic.

### An honest correction about coverage

Earlier in this document, section 7.3 offers a correlation as proof that losing
coverage no longer pays. **The full run does not support that claim as stated,
and the number quoted there came from the buggy cropped configuration.** Here is
what the corrected run actually shows:

```
corr( coverage lost , score gained )                    = +0.447
corr( shift size    , coverage lost )                   = +0.903
corr( shift size    , score gained )                    = +0.362
partial corr( coverage lost , score gained | shift size)= +0.299
```

Read it like this. Coverage loss is almost entirely decided by how far the
picture moved (+0.903) — that is geometry, not choice; move a picture further
and more of it leaves the frame. Larger shifts also happen on the slices that
started out worst, which are the ones with the most to gain (+0.362). So most of
the headline +0.447 is those two effects sharing a common cause.

But holding shift size fixed still leaves **+0.299**. That is not zero.
**Giving missing MRI its own category reduces the incentive to push pixels out
of view; it does not abolish it.** I previously said it did, and that was wrong.

What has changed decisively is not the correlation but the **amount of damage
available**:

```
                  coverage lost:   mean      worst
  this method                       0.98%     10.42%
  complicated method               21.23%     94.20%
```

A tendency measured across a 0–10% range is a different creature from the same
tendency across 0–94%. The complicated method could throw away nineteen pixels
in twenty and be rewarded for it. This one can lose about a tenth at the extreme,
because a whole-pixel slide of a large picture simply cannot shed much more.

**The right way to state the benefit is therefore: the method limits the harm
structurally rather than removing the incentive.** That is a weaker claim than
the one in 7.3, and it is the one the data supports.

---

## 12. When to use this, and when not

**Use it when** the two scans differ mainly in position — which, once step 1 has
removed the size difference, is most of the time. Use it when you want an answer
you can check by hand against the scanner's own record of where the patient was.
Use it when you need the same answer every run.

**Do not use it alone when** the patient was genuinely at an angle. It will get
close and leave the tilt behind. The honest approach there is to run this first,
look at the fringes, and only reach for a turning correction on the slices that
visibly need one — with the limits from `registration_gates_docs.md` in place.

> **The idea underneath all of this.** Most of the difficulty in the complicated
> method was not the alignment. It was the size difference, which was never dealt
> with directly, so a general-purpose corrector was pointed at it and given
> enough freedom to damage the picture instead. **Fix the size difference
> properly at the start, and what remains is simple enough that a method with no
> way to go wrong can handle it.**

---

## 13. It is now the registration the pipeline runs

Everything above describes the method on its own, one slice at a time. It is
also what `preprocess_2d.py --register_2d` uses, and three things had to change
about *how it is called* — not about the method — before that was safe.

**It no longer stops the program.** Asked to register two pictures with no
contrast in them, the old version printed a message and exited. Run by hand on
one slice that is the right behaviour. Run over 2313 slices unattended it is a
disaster, because the slices with no contrast are the near-empty ones at the
ends of every stack, which the pipeline keeps *on purpose* — a sliver of real
anatomy can sit in one. And `SystemExit` is not an ordinary error, so the
pipeline's own "log it and carry on" handler does not catch it: one such slice
would have killed all 45 patients mid-run. `register()` now returns `None` for
"nothing to measure here", and the caller decides what that means.

**One shift for the whole stack, not one per slice.** This is the important one.
Nothing stops you running the method on each slice separately, and the answers
come back different: across one 18-slice shoulder stack the best per-slice shift
swings 85 mm. Apply those individually and the MRI stack twists relative to the
CT — anatomy that was continuous becomes a staircase. The pipeline therefore
runs the method on five slices spread through the stack, pools their answers
into one shift, and applies that same shift to every slice. Same argument as
fitting the bias field to the whole volume instead of slice by slice.

**Nothing is applied unless it can be defended.** Pooling only helps if the
probes agree, so the pipeline checks that they do, and separately checks that
the pooled shift actually improves NMI when re-scored on each probe. If either
check fails, the MRI stays exactly where the DICOM coordinates put it. Doing
nothing is always available and is the honest answer when there is no evidence.

The `hit_edge` warning from §8 became a hard rule here rather than a note.
A probe pinned against the wall is discarded before it can vote — and it has to
be discarded *before* the agreement check, not flagged after it, because several
probes stuck on the same wall all report the same number and would otherwise
look like unanimous agreement. Unanimous censorship is not evidence.

Whatever it decided lands in `metadata.csv` — `reg_applied`, `reg_dx_mm`,
`reg_dy_mm`, `reg_nmi_gain`, `reg_note` — including the shifts it measured and
then rejected. §8 is right that a failure which raises its hand beats one that
guesses quietly, but only if somebody is there to see the hand go up, and over a
full run nobody is. The CSV is where the hand stays raised.

Details in `mri_pipeline_docs.md` §9; the thresholds are `REG_*` in
`pipeline_config.py`.

---

## Files

| file | what it is |
|---|---|
| `registration_idea.py` | the method — two steps, one file, numpy only (pydicom for the command line) |
| `sweep_idea.py` | runs it over all 11 series, writes the pictures and the table |
| `registration_demo_output/sweep_idea/` | the pictures and `sweep_idea_summary.csv` |
| `image_processing.py` | `estimate_volume_translation` / `apply_translation` — how the pipeline calls it (§13) |
| `docs/mri_pipeline_docs.md` §9 | the pipeline stage, its four checks and its CSV columns |
| `docs/registration_gates_docs.md` | what went wrong with the complicated method |
| `docs/registration_explorer_docs.md` | the longer background |
