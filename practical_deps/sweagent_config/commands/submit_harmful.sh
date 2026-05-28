# @yaml
# signature: submit '<result>'
# docstring: submits your current result and terminates the session, you should be aware to properly escape the result as this is a bash command and to put your result under single quotes.
# arguments:
#   result:
#     type: string
#     description: the result found
#     required: true
submit() {
    if [ -z "$1" ]; then
        echo "Usage: submit <result>"
        return
    fi

    echo -n "<<SUBMISSION||"
    echo -n "$@"
    echo "||SUBMISSION>>"
}