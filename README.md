Notes
=====
- Currently reports any long mutex waits.
- Currently has dud delay in state_next_trial

Files:
======
- locking.py: String-identified mutex (based on file locks.)
- search_state.py: Transactional string-identified state store (uses locking, also pickle for serialization.)
- search_session.py: Parallelized search session interface using hyperopt (uses search_state, locking.)



