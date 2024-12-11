# Copyright 2024 The JAX SC Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dataset for shakespeare next word predictor."""

import collections
from absl import logging
import numpy as np

_SHAKESPEARE = """
  Let those who are in favour with their stars,
  Of public honour and proud titles boast,
  Whilst I whom fortune of such triumph bars
  Unlooked for joy in that I honour most;
  Great princes' favourites their fair leaves spread,
  But as the marigold at the sun's eye,
  And in themselves their pride lies buried,
  For at a frown they in their glory die.
  The painful warrior famoused for fight,
  After a thousand victories once foiled,
  Is from the book of honour razed quite,
  And all the rest forgot for which he toiled:
    Then happy I that love and am beloved
    Where I may not remove nor be removed.

  Lord of my love, to whom in vassalage
  Thy merit hath my duty strongly knit;
  To thee I send this written embassage
  To witness duty, not to show my wit.
  Duty so great, which wit so poor as mine
  May make seem bare, in wanting words to show it;
  But that I hope some good conceit of thine
  In thy soul's thought (all naked) will bestow it:
  Till whatsoever star that guides my moving,
  Points on me graciously with fair aspect,
  And puts apparel on my tattered loving,
  To show me worthy of thy sweet respect,
    Then may I dare to boast how I do love thee,
    Till then, not show my head where thou mayst prove me.

  Weary with toil, I haste me to my bed,
  The dear respose for limbs with travel tired,
  But then begins a journey in my head
  To work my mind, when body's work's expired.
  For then my thoughts (from far where I abide)
  Intend a zealous pilgrimage to thee,
  And keep my drooping eyelids open wide,
  Looking on darkness which the blind do see.
  Save that my soul's imaginary sight
  Presents thy shadow to my sightless view,
  Which like a jewel (hung in ghastly night)
  Makes black night beauteous, and her old face new.
    Lo thus by day my limbs, by night my mind,
    For thee, and for my self, no quiet find.

  How can I then return in happy plight
  That am debarred the benefit of rest?
  When day's oppression is not eased by night,
  But day by night and night by day oppressed.
  And each (though enemies to either's reign)
  Do in consent shake hands to torture me,
  The one by toil, the other to complain
  How far I toil, still farther off from thee.
  I tell the day to please him thou art bright,
  And dost him grace when clouds do blot the heaven:
  So flatter I the swart-complexioned night,
  When sparkling stars twire not thou gild'st the even.
    But day doth daily draw my sorrows longer,
    And night doth nightly make grief's length seem stronger
"""


def load_shakespeare(vocab_size: int) -> list[int]:
  """Loads a shakespeare dataset and converts it to token IDs.

  Splits the input data into words, and assigns the top `vocab_size` words a
  unique ID in the range 0..vocab_size. Returns a list of word IDs, with words
  not in the top VOCAB_SIZE words represented as ID 0.

  Args:
    vocab_size: The number of words to use for the embedding table.

  Returns:
    list of word IDs
  """
  logging.info('Loading shakespeare')
  words = _SHAKESPEARE.split()
  words = [w.strip() for w in words]
  counts = collections.defaultdict(int)
  logging.info('Counting words')
  for w in words:
    counts[w] += 1
  top_n = list(sorted(list(counts.items()), key=lambda v: v[1], reverse=True))
  top_n = top_n[: vocab_size]
  logging.info('Top 5: %s', top_n[:5])
  word_to_id = {}
  for word, _ in top_n:
    word_to_id[word] = len(word_to_id)

  word_ids = []
  for w in words:
    word_ids.append(word_to_id.get(w, 0))
  logging.info('Loaded %d words', len(word_ids))
  return word_ids


# Our goal is to predict the next word from the current SEQ_LEN words.
# We iterate sequentially through our dataset. To simplify convergence testing,
# we use a single batch.
def word_id_batches(word_ids, num_steps, batch_size, seq_len, num_tables):
  """Generates the feature and label batches.

  Args:
    word_ids: The text mapped to integer ids.
    num_steps: The total number of steps
    batch_size: Local batch size.
    seq_len: The size of the sequences used to predict the next word.
    num_tables: The total number of embedding tables.

  Returns:
    Tuple of features/labels.
  """
  word_tensors = []
  label_tensors = []
  for _ in range(num_steps):
    start = 0
    end = batch_size

    # Each domain will map to [idx, idx + SEQ_LEN.value - 1]. Note that the
    # next word for each domain is then idx + SEQ_LEN.value.
    x_idx = np.arange(start, end, dtype=np.int32) % len(word_ids)
    y_idx = np.arange(
        start + seq_len, end + seq_len, dtype=np.int32
    ) % len(word_ids)

    words = np.array([word_ids[idx : idx + seq_len] for idx in x_idx])
    words = np.expand_dims(words, -1)
    word_tensors.append(words.astype(np.int32))

    labels = np.array([word_ids[idx] for idx in y_idx])
    label_tensors.append(labels.astype(np.int32))

  features = {}
  for i in range(num_tables):
    features['words_%d' % i] = word_tensors

  return features, label_tensors
