# Bash completion for qmd
# Source this file or copy to /etc/bash_completion.d/qmd

_qmd() {
  local cur prev words cword
  _init_completion || return

  local commands="collection context ls get multi-get status update embed pull search vsearch query mcp cleanup"
  local collection_sub="add list remove rm rename mv"
  local context_sub="add list check rm remove"
  local mcp_sub="stop"

  local global_opts="--index --help -h"
  local search_opts="-n --min-score --all --full --line-numbers -c --collection --csv --md --xml --files --json"
  local get_opts="-l --from --line-numbers"
  local multiget_opts="-l --max-bytes --line-numbers --csv --md --xml --files --json"
  local embed_opts="-f --force"
  local update_opts="--pull"
  local mcp_opts="--http --daemon --port"
  local pull_opts="--refresh"
  local collection_add_opts="--name --mask"

  # Find the main command (first positional)
  local cmd="" subcmd="" i
  for ((i = 1; i < cword; i++)); do
    case "${words[i]}" in
      --*) ;; # skip options
      -*)  ;; # skip short options
      *)
        if [[ -z "$cmd" ]]; then
          cmd="${words[i]}"
        elif [[ -z "$subcmd" ]]; then
          subcmd="${words[i]}"
        fi
        ;;
    esac
  done

  # Complete option values
  case "$prev" in
    --index|--name|--mask|--port)
      return 0
      ;;
    -n|-l|--min-score|--max-bytes|--from)
      return 0
      ;;
    -c|--collection)
      # Complete collection names from qmd collection list
      local collections
      collections=$(qmd collection list 2>/dev/null | awk '/\(qmd:\/\// {print $1}')
      COMPREPLY=($(compgen -W "$collections" -- "$cur"))
      return 0
      ;;
  esac

  case "$cmd" in
    "")
      # Complete main commands + global opts
      COMPREPLY=($(compgen -W "$commands $global_opts" -- "$cur"))
      return 0
      ;;
    collection)
      case "$subcmd" in
        "")
          COMPREPLY=($(compgen -W "$collection_sub" -- "$cur"))
          return 0
          ;;
        add)
          if [[ "$cur" == -* ]]; then
            COMPREPLY=($(compgen -W "$collection_add_opts" -- "$cur"))
          else
            _filedir -d
          fi
          return 0
          ;;
        remove|rm|rename|mv)
          local collections
          collections=$(qmd collection list 2>/dev/null | awk '/\(qmd:\/\// {print $1}')
          COMPREPLY=($(compgen -W "$collections" -- "$cur"))
          return 0
          ;;
      esac
      ;;
    context)
      case "$subcmd" in
        "")
          COMPREPLY=($(compgen -W "$context_sub" -- "$cur"))
          return 0
          ;;
        rm|remove)
          # Could complete known context paths
          return 0
          ;;
      esac
      ;;
    ls)
      # Complete collection names, then paths within
      local collections
      collections=$(qmd collection list 2>/dev/null | awk '/\(qmd:\/\// {print $1}')
      COMPREPLY=($(compgen -W "$collections" -- "$cur"))
      return 0
      ;;
    get)
      if [[ "$cur" == -* ]]; then
        COMPREPLY=($(compgen -W "$get_opts" -- "$cur"))
      fi
      return 0
      ;;
    multi-get)
      if [[ "$cur" == -* ]]; then
        COMPREPLY=($(compgen -W "$multiget_opts" -- "$cur"))
      fi
      return 0
      ;;
    search|query|deep-search)
      if [[ "$cur" == -* ]]; then
        COMPREPLY=($(compgen -W "$search_opts" -- "$cur"))
      fi
      return 0
      ;;
    vsearch|vector-search)
      if [[ "$cur" == -* ]]; then
        COMPREPLY=($(compgen -W "$search_opts" -- "$cur"))
      fi
      return 0
      ;;
    embed)
      if [[ "$cur" == -* ]]; then
        COMPREPLY=($(compgen -W "$embed_opts" -- "$cur"))
      fi
      return 0
      ;;
    update)
      if [[ "$cur" == -* ]]; then
        COMPREPLY=($(compgen -W "$update_opts" -- "$cur"))
      fi
      return 0
      ;;
    pull)
      if [[ "$cur" == -* ]]; then
        COMPREPLY=($(compgen -W "$pull_opts" -- "$cur"))
      fi
      return 0
      ;;
    mcp)
      case "$subcmd" in
        "")
          if [[ "$cur" == -* ]]; then
            COMPREPLY=($(compgen -W "$mcp_opts" -- "$cur"))
          else
            COMPREPLY=($(compgen -W "stop $mcp_opts" -- "$cur"))
          fi
          return 0
          ;;
      esac
      ;;
    status|cleanup)
      return 0
      ;;
  esac
}

complete -F _qmd qmd
