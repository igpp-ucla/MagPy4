''' Interactive prompting with injectable input/output for testing '''
import sys

class CLIError(Exception):
    ''' A user-facing error; the message is printed without a traceback '''

class Prompter():
    def __init__(self, interactive=True, input_fn=None, out=None):
        self.interactive = interactive
        self.input_fn = input_fn if input_fn is not None else input
        self.out = out if out is not None else sys.stdout

    def say(self, text=''):
        print(text, file=self.out)

    def ask(self, text, default=None, parse=None, flag=None, allow_empty=False):
        ''' Asks for a value, re-asking until parse(text) succeeds. An empty
            answer gives the default, or '' if allow_empty is set.
            Raises a CLIError naming the flag when not interactive.
            Prompts are written to self.out (stderr for the real CLI) so
            that results on stdout stay clean.
        '''
        if not self.interactive:
            hint = f' (pass {flag})' if flag else ''
            raise CLIError(f'Missing required value: {text}{hint}')

        suffix = f' [{default}]' if default not in (None, '') else ''
        while True:
            self.out.write(f'{text}{suffix}: ')
            self.out.flush()
            try:
                answer = self.input_fn()
            except EOFError:
                raise CLIError('Input closed while waiting for: ' + text)
            answer = answer.strip()
            if answer == '':
                if default is None and allow_empty:
                    return ''
                if default is None:
                    self.say('A value is required.')
                    continue
                answer = default
            if parse is None or not isinstance(answer, str):
                return answer
            try:
                return parse(answer)
            except (ValueError, CLIError) as e:
                self.say(f'Invalid input: {e}')

    def choose(self, text, options, default=None, flag=None):
        ''' Asks for one of options (by name or number) '''
        if self.interactive:
            self.say(text)
            for i, opt in enumerate(options, 1):
                label = opt if isinstance(opt, str) else opt[0]
                note = '' if isinstance(opt, str) or not opt[1] else f' - {opt[1]}'
                self.say(f'  {i}) {label}{note}')
        names = [o if isinstance(o, str) else o[0] for o in options]

        def parse(answer):
            if answer in names:
                return answer
            if answer.isdigit() and 1 <= int(answer) <= len(names):
                return names[int(answer) - 1]
            raise ValueError(f"choose one of: {', '.join(names)}")

        label = 'Choice' if self.interactive else text
        return self.ask(label, default=default, parse=parse, flag=flag)

    def confirm(self, text, default=False):
        ''' Yes/no question; returns the default when not interactive '''
        if not self.interactive:
            return default
        tag = 'Y/n' if default else 'y/N'

        def parse(answer):
            if answer.lower() in ('y', 'yes'):
                return True
            if answer.lower() in ('n', 'no'):
                return False
            raise ValueError("answer y or n")

        return self.ask(f'{text} ({tag})', default='y' if default else 'n',
            parse=parse)
