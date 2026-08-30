# Chapter 4 Canonical SQL Reference
## SQL QuickStart Guide (ch. 4 "Getting Started with Queries", pp. 51–62) ↔ wsda_music.db

This file is the canonical content source for the five chapter-4 videos.
LessonBuilder transcribes this SQL verbatim; grounding asserts these expected
results; the editor format contract below is the on-screen standard.

## Editor Format Contract ("tight" standard)

1. Content begins at **line 1** — zero leading blank lines.
2. The comment block is **immediately** followed by its query — no blank line between.
3. Fields indented exactly **2 spaces**; no tabs anywhere; no trailing whitespace.
4. History queries from prior videos: every line commented with `--`, then one
   blank line, then the current block. Current block is never commented.
5. No literal escape sequences (`\t`, `\n`) may ever appear as visible text.
6. One statement per block, terminated with `;`.

## Video 1 — Your First Query
Objective: write and run a first SELECT returning a customer contact list.

```sql
/*
Created By: WSDA Student
Create Date: {record date}
Description: Customer contact list for management
*/
SELECT
  FirstName,
  LastName,
  Email
FROM Customer;
```
Expected result: **60 rows**; first row **Luís Gonçalves**; columns FirstName, LastName, Email.
(Book equivalent: sTunes `customers` table, 59 rows — our wsda_music.db has 60.)

## Video 2 — Aliases (AS)
Objective: readable column headers via AS.

```sql
/*
Created By: WSDA Student
Create Date: {record date}
Description: Readable customer contact headers
*/
SELECT
  FirstName AS "First Name",
  LastName AS "Last Name",
  Email AS "Email Address"
FROM Customer;
```
Expected result: **60 rows**; headers exactly `First Name | Last Name | Email Address`.
(Book uses bracket style `AS [First Name]`; double quotes are the SQL standard —
either is valid, but be consistent within a course.)

## Video 3 — ORDER BY
Objective: sort the contact list alphabetically by last name.

```sql
/*
Created By: WSDA Student
Create Date: {record date}
Description: Customer contact list sorted by last name
*/
SELECT
  FirstName AS "First Name",
  LastName AS "Last Name",
  Email AS "Email Address"
FROM Customer
ORDER BY LastName;
```
Expected result: **60 rows**; first row **Roberto Almeida** (roberto.almeida@riotur.gov.br).
Teaching point (from the book): without ORDER BY, rows return in stored order.

## Video 4 — LIMIT
Objective: return only a small preview of the sorted list.

```sql
/*
Created By: WSDA Student
Create Date: {record date}
Description: Preview of customer contacts
*/
SELECT
  FirstName AS "First Name",
  LastName AS "Last Name",
  Email AS "Email Address"
FROM Customer
ORDER BY LastName
LIMIT 5;
```
Expected result: **5 rows**; first row Roberto Almeida.
(Book demonstrates "top ten" with LIMIT 10; this course uses 5.)

## Video 5 — Etiquette Recap
Objective: one clean, documented, professional statement combining header
comment, aliases, ORDER BY, LIMIT — the Video 4 query presented as the
finished standard, with the full history visible above it, commented out.

```sql
/*
Created By: WSDA Student
Create Date: {record date}
Description: Clean, documented customer contact preview
*/
SELECT
  FirstName AS "First Name",
  LastName AS "Last Name",
  Email AS "Email Address"
FROM Customer
ORDER BY LastName
LIMIT 5;
```
Expected result: **5 rows**; first row Roberto Almeida.

## Checkpoint questions (from the book — validation/close beat material)
- Add the Company or Phone field to the mailing-list query. (Don't forget the comma.)
- Rearrange SELECT so LastName comes first, order by LastName — is it more readable?
- What changes between LIMIT 5 and LIMIT 10?

## Grounding expectations summary (for the must-fail grounding check)
| Video | Rows | First row | Headers |
|-------|------|-----------|---------|
| 1 | 60 | Luís Gonçalves | FirstName, LastName, Email |
| 2 | 60 | Luís Gonçalves | First Name, Last Name, Email Address |
| 3 | 60 | Roberto Almeida | First Name, Last Name, Email Address |
| 4 | 5 | Roberto Almeida | First Name, Last Name, Email Address |
| 5 | 5 | Roberto Almeida | First Name, Last Name, Email Address |
