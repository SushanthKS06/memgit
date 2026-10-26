## subject
memory session {session_id}: {counts}

## count_format
+{added} ~{superseded} -{rejected} (noop {noop})

## added_line
- learned [{id}] {fact} (t{turn}, s:{session})

## superseded_line
- updated [{new_id}] {fact} (t{turn}; was [{old_id}] {old_fact})

## rejected_line
- rejected: {reason}

## empty_body
(nothing stored this session)

## body
{operations}
