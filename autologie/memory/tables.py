""" 
conversations
    user | agent | datetime | role | content | images | objects | tool_calls | tool_responses 
files
    id | file-name | location | size | created-at | modified-at | indexed_at | contents-string | vector | summary
chunks
    id | contents | file-id | chunk-index | summary | parent-id | child-id | vector
scraped-webpages
    id | url | created-at | html | text | json-objcts | visual-browsing-content | summary
agent-memories
    id | agent-id | created-at | memory-string | user | conversation-id/source_message/source_conversation
vectors
    id
"""