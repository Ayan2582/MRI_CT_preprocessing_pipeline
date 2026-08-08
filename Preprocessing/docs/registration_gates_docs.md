# The gates: what they are, why the current one misses things, and what to add

*Written to be read start to finish with no prior background. Every number in it
was measured on this project's own data — the 33-slice `sweep_og` run.*

---

## 1. What a gate is

The registration step does not produce one answer. It produces several
candidates, and then something has to choose.

The choosing happens in two separate parts, and keeping them separate is the
whole point:

- **The score** asks *"which candidate lines the two pictures up best?"*
- **The gate** asks *"which candidates are we even willing to consider?"*

A gate does not rank anything. It only says yes or no. A candidate that fails
the gate is thrown out no matter how well it scored.

Why have one at all? Because the score can be fooled. We will see three ways it
gets fooled, all of them happening right now in this project. If the score could
be trusted completely, you would just take the highest one and go home. The gate
exists because it cannot.

> **One sentence to keep:** the score says *how good*, the gate says *how
> allowed*.

---

## 2. What we are choosing between

The registration tries to find a way of moving the MRI so it sits on top of the
CT. "A way of moving" is described by six numbers:

- two of them slide the picture sideways and up/down,
- the other four are arranged in a little square block of numbers called a
  matrix, and between them they can turn the picture, make it bigger or smaller,
  and skew it.

The two sliding numbers are harmless. Nobody worries about them; sliding a
picture cannot damage it. **All the trouble is in the four-number block.** So
that is what the gate looks at.

---

## 3. How to read a four-number block

This is the one piece of maths in the document, and there is a picture that
makes it easy.

Draw a circle. Feed it through the four-number block. **What comes out is always
an ellipse** — a squashed circle. Always. There is no set of four numbers that
turns a circle into anything else.

So any four-number block can be described by just three plain facts about the
ellipse it produces:

```
        ○  circle in                    ⬭  ellipse out
                                        │
                          long radius ──┤
                                        │  ← short radius
```

| Fact about the ellipse | What it tells you |
|---|---|
| **long radius** (call it `s₁`) | how much it stretched the most |
| **short radius** (call it `s₂`) | how much it stretched the least |
| **how far the ellipse is turned** | how much it rotated the picture |

Now the three useful measurements, all built from those:

| Name | How to work it out | What it means in words |
|---|---|---|
| **size** | `(s₁ + s₂) / 2` | the average radius — is the picture bigger or smaller overall? |
| **turn** | the angle the ellipse sits at | how far the picture was rotated |
| **imbalance** | `s₁ / s₂` | how far from a circle it is — 1 means still a circle, big means a long thin ellipse |

Keep those three words: **size**, **turn**, **imbalance**. The rest of the
document is about the fact that we currently measure only the first one.

---

## 4. What the gate does today

Today's gate looks at **size** and nothing else. It runs two tests.

### Test 1 — the sanity bound

> Reject if the size is more than 0.60 away from 1.

A size of 1 means "same size as before". So this allows anything from a picture
shrunk to 40% up to one blown up to 160%. That is deliberately generous. Two
scans of the same person should not differ in size by anything like that, so
this test is there to catch outright nonsense, not to be fussy.

### Test 2 — the frame-matching test

This one catches a specific trick, and it is worth understanding because it is
the same *kind* of problem as everything else in this document.

The MRI does not fill its frame. There is a rectangle of real picture and black
nothing around it. The frame itself is also a rectangle. Both rectangles have
long straight edges, and a straight edge is a very strong feature.

So the machine discovers it can raise its score by **stretching the MRI until
its edge sits exactly on the frame's edge**. Two strong straight lines now agree.
The score goes up. Not one piece of anatomy is any better aligned.

You can predict the exact size it will pick: whatever ratio makes the two edges
meet. If the MRI's picture covers 66% of the frame, the machine will land on a
size of about 0.66, or its flip side 1.51. So the test is:

> Work out that suspicious ratio in advance. Reject any candidate whose size
> lands within 0.04 of it, or of its flip side.

There is a guard on this test: if the MRI already fills its frame, there are no
two edges to bring together, and a size near 1 is simply the right answer rather
than a suspicious one. So the test only runs when the MRI genuinely falls short
of the frame.

**This is a good test.** It is well reasoned and it works. The problem is not
that it is wrong. The problem is that it is the only one.

---

## 5. The first hole: turning is invisible

Go back to the circle and the ellipse.

**A rotation turns a circle into a circle.** It moves it round, but it does not
squash it. So both radii are still exactly 1.

Which means:

```
size = (1 + 1) / 2 = 1.0000
```

Not approximately. Exactly. **For every angle.**

Measured on a 2×2 rotation, which is what the code actually computes:

```
  1 degree  ->  size = 1.0000
 10 degrees ->  size = 1.0000
 30 degrees ->  size = 1.0000
 90 degrees ->  size = 1.0000
```

A picture turned completely on its side reports exactly the same number as one
left untouched.

> **There is no threshold you could ever set on `size` that would catch a
> rotation.** It is not a matter of picking a better number. A rotation does not
> change the radii, and `size` is made only of the radii. The information is not
> there to be found.

The gate is not being careless here. It is being asked a question it cannot
answer.

### What that let through

In the 33-slice run, two slices shipped a rotation of about 21 degrees:

```
knee  sagittal middle   turned -21.3 degrees   score went UP by 0.1185
spine coronal  middle   turned +20.9 degrees   score went UP by 0.1095
```

Both reported `size = 1.0000`. Both sailed through.

Look at `knee_PA32_Mandbi_knee_sagittal_middle_og.png` and you can see the
result without needing any of this explained: the "before" picture shows a knee
with both scans sitting neatly on top of each other. The "after" picture is
nearly empty. The MRI has been turned so far that almost all of it fell off the
edge of the frame.

**The starting position was good. The machine made it much worse, and the score
went up.**

---

## 6. The second hole: skew is nearly invisible

Skew — pushing the top of a picture sideways while the bottom stays put, turning
a square into a leaning parallelogram — is worse, because it *looks* like
something `size` ought to notice.

Here is why it does not. A skew that leaves the area unchanged makes the circle
into an ellipse that is **longer in one direction and thinner in the other, by
matching amounts**. The long radius goes up. The short radius goes down. Average
them and they nearly cancel out.

Measured, with `k` being how hard the skew pushes:

| skew | how far it leans | **size** says | gate says | **imbalance** says |
|---|---|---|---|---|
| 0.14 | 8° | 1.0024 | allowed | 1.150 |
| 0.30 | 17° | 1.0112 | allowed | 1.348 |
| 0.50 | 27° | 1.0308 | allowed | 1.640 |
| 2.50 | 68° | 1.6008 | **rejected** | 8.127 |

Read across the 27° row. The picture is visibly leaning over. `size` reports
1.03 — a three percent change. The limit is 0.60. It is not remotely close to
being caught. You would need to lean the picture **68 degrees** before the
current gate objects.

Now read the last column. `imbalance` goes 1.15, 1.35, 1.64, 8.13. It climbs
steadily exactly where `size` sits flat. **The information was always there. We
were looking at the wrong number.**

### What that let through

```
spine coronal first    imbalance 14.46    size 0.602    vetoed: none    "improved"
```

An imbalance of 14.46 means the picture came out fourteen times longer than it
is wide. `size` reported 0.602 — which is 0.398 away from 1, comfortably inside
the limit of 0.60 — so the gate rejected nothing.

Open `spine_PA18_Sangeeta_coronal_first_og.png`. The MRI is not a picture of a
spine any more. It is a set of diagonal stripes, like a smear. Every piece of
anatomy is gone.

**That slice has the biggest score improvement in the whole run: +0.1555.** The
pipeline labelled it "improved" and shipped it.

Across all 33 slices, **12 shipped with an imbalance above 1.15**, and the gate
fired on 7 candidates out of 99.

---

## 7. The third hole: the score pays for deleting things

This one is not about the four-number block at all. It is about the score
itself, and it is the deepest of the three.

The score is only worked out on pixels where **both** pictures have real data.
Anywhere the MRI has nothing — outside its edge, or off the side of the frame —
that pixel is skipped.

Now think about what happens when you turn or stretch the MRI. Corners swing off
the edge. Those pixels stop counting. **So changing the transform changes which
pixels are being marked.**

That is like being allowed to cross out the exam questions you got wrong before
handing the paper in. Your percentage goes up. You did not learn anything.

And it is worse than random, because of *which* pixels get crossed out first.
The corners and edges are the parts least likely to have a matching structure in
the CT — they are the pixels dragging the score down. **Throwing away your
worst-fitting pixels makes the score rise.** The machine has found that out.

There is meant to be a safeguard: a candidate is refused if fewer than 5% of
pixels are left. Five percent. A candidate can throw away nineteen pixels in
every twenty and still be scored as if nothing happened.

### What that let through

Both 21-degree rotations went from **100% of pixels down to 6%** — and squeaked
past the 5% floor by one point.

Across the run:

```
correlation between "how much was thrown away" and "how much the score rose" = +0.51

  slices that threw away more than 30%:   7 slices, average score gain +0.0919
  slices that threw away less than 30%:  26 slices, average score gain +0.0603
```

**The more a transform destroyed, the more the score went up.** That is not a
coincidence in a few bad slices; it is the pattern across the whole run.

One fair qualification: losing a *little* is normal and innocent. Slide a picture
ten pixels and a thin strip falls off the edge. Slices sitting at 90–97% are
fine. The ones that matter are the tail: 6%, 6%, 24%, 43%, and five more below
65%.

---

## 8. The three gates to add

Each one plugs exactly one hole. None of them replaces the existing gate — the
size test and the frame-matching test both stay.

### Gate A — a limit on turning

> Reject any candidate that turns the picture by more than **12 degrees**.

**Why 12.** Both pictures are of the same person, lying down, positioned by a
radiographer, in the same convention. The difference between the two is somebody
shifting on a table between appointments. That is a few degrees. Twelve is
already generous.

If your data really needs more than that, raise it — but raise it deliberately
and write down the reason. The point is not that 12 is sacred. The point is that
**"no limit at all" is not a decision anybody made**; it is what you get by
accident when the only thing you measure cannot see rotation.

> **Apply this to the simple transform too.** At the moment the simple
> (slide-and-turn-only) transform skips the gate entirely, with the reasoning
> "it cannot change size, so there is nothing to check." That reasoning is
> correct about size and silent about turning — and **two of the three worst
> slices in the run were the simple transform**, at 21 degrees. It turns just as
> freely as the complicated one. The size and imbalance tests will do nothing to
> it, so only the turning limit will ever bite. Let it.

### Gate B — a limit on imbalance

> Reject any candidate whose imbalance (long radius ÷ short radius) is above
> **1.15**.

**Why 1.15.** From the measured table above, an imbalance of exactly 1.150 is
what an 8-degree lean produces. Eight degrees of lean is about as much as a
person lying slightly differently can explain. So the number was not picked out
of the air — it was read off the point where real skew begins.

This is the gate that catches the stripe disaster. That candidate had an
imbalance of 14.46 against a limit of 1.15.

### Gate C — a floor on how much may be thrown away

> Reject any candidate that keeps less than **90% of the pixels the untouched
> starting position kept**.

Two things changed from the current rule:

1. **It is measured against the starting position, not against zero.** "Keep at
   least 5% of the frame" is meaningless when the starting position already
   keeps 100%. "Do not lose more than a tenth of what you started with" means
   something at any starting point.
2. **The number is 90%, not 5%.** Losing a thin strip off the edge is fine.
   Losing more than a tenth means the candidate is winning by deletion.

This is a patch, not a cure. The real cure is to decide **once**, before any
optimising starts, which pixels count — and then score every candidate on that
same fixed set, treating a candidate that pushes anatomy out of it as having
missed. Then no candidate can improve its score by shrinking the exam. Gate C is
the cheap version that gets most of the benefit; the fixed-set version is the
right one and is a bigger change.

---

## 9. All three come free

Here is the part that makes this easy to accept: **you are already computing
everything you need.**

The existing size calculation works out four intermediate quantities on its way
to the answer, and then throws three of them away. Keep them and you get all
three new measurements at no extra cost:

```python
def decompose(M):
    a, b, c, d = M[0,0], M[0,1], M[1,0], M[1,1]
    E, F, G, H = (a+d)/2, (a-d)/2, (c+b)/2, (c-b)/2      # already computed today
    Q, R = hypot(E, H), hypot(F, G)                       # already computed today
    s1, s2 = Q + R, abs(Q - R)                            # the two radii
    return {
        "size":       (s1 + s2) / 2,      # what the gate tests now
        "turn":       atan2(H, E),        # NEW — radians
        "imbalance":  s1 / s2,            # NEW — 1.0 means "still a circle"
    }
```

No new loops, no new passes over the image, nothing slower. Three lines.

The gate then becomes:

```python
def verdict(M, frame_ratio, coverage, coverage_at_start):
    D = decompose(M)

    if abs(D["size"] - 1) > 0.60:                    return False, "too big or small"
    if abs(degrees(D["turn"])) > 12:                 return False, "turned too far"
    if D["imbalance"] > 1.15:                        return False, "skewed"
    if coverage < 0.90 * coverage_at_start:          return False, "threw too much away"

    if 0.05 < frame_ratio < 0.95:                    # the existing frame test, unchanged
        for suspect in (frame_ratio, 1/frame_ratio):
            if abs(D["size"] - suspect) <= 0.04:     return False, "matched the frame edge"

    return True, ""
```

**Check it before believing it.** Feed in a known 10-degree rotation: you should
get `turn = 10.00`, `size = 1.0000`, `imbalance = 1.0000`. Feed in a known skew
of 0.30: you should get `imbalance = 1.348`. If those two do not come out right,
nothing built on top means anything.

---

## 10. What these would have caught

Applied to the 33 slices that were actually run:

| Gate | Slices it stops |
|---|---|
| A — turning above 12° | **3** — including both 21° cases |
| B — imbalance above 1.15 | **12** — including the 14.46 stripe disaster |
| C — kept under 90% of the start | **20** |
| **Any of the three** | **24 of 33** |

Only **9 of 33** slices pass all three cleanly today.

That number should be uncomfortable, and it should also be treated carefully.
Twenty-four failures does not mean twenty-four ruined pictures — Gate C is the
sensitive one, and the rows sitting at 90–95% are minor. The severe cases are
the handful in the tail. But nine out of thirty-three passing every check is a
long way from where a pipeline you would trust should sit.

---

## 11. How to tell if you have gone too far

A gate that rejects everything is not safe, it is useless. So check all four of
these after turning them on:

1. **Does the honest test still pass?** A known 10-degree rotation must give
   `turn = 10.00`, `size = 1.0000`, `imbalance = 1.0000`. Start here every time.

2. **Do the three bad slices now get refused?** `knee/sagittal/middle`,
   `spine/coronal/middle`, `spine/coronal/first`. If they still ship, something
   is not wired in.

3. **Did anything start going backwards?** Run the full sweep and look for the
   result "worse than doing nothing". There are **none** today. If some appear,
   the gates have become tight enough to throw away real corrections, and the
   thresholds need loosening — with a note saying why.

4. **Do the scores go down?** They should, on the bad slices, and that is the
   sign of success rather than failure. The big numbers were bought by damaging
   the picture. A smaller honest number is worth more than a larger dishonest
   one.

> **The one idea underneath all of this.** A score tells you how well two
> pictures agree *on the pixels you chose to compare*. It cannot tell you whether
> the transform that got you there was a sensible thing to do to a human body.
> That second question has to be asked separately, by something that looks at
> the transform itself. That is what a gate is for, and why one measurement was
> never going to be enough.

---

## Where things live

| File | What it is |
|---|---|
| `registration_og.py` | the v3 pipeline; already has `decompose()` |
| `working_regis.py` | the v6 pipeline |
| `registration_demo_sweep_v3.py` | the production pipeline — same blind spot, in `affine_scale()` |
| `sweep_og.py` | the 33-slice run all the numbers here come from |
| `registration_demo_output/sweep_og/` | the pictures, plus `sweep_og_summary.csv` |
| `docs/registration_explorer_docs.md` | the longer companion, with the v3/v6 comparison |
