import uuid
import nbformat as nbf


metadata = {
    "kernelspec": {"display_name": "tfod",
                   "language": "python",
                   "name": "tfod"
                   },
    "language_info": {"codemirror_mode": {"name": "ipython",
                                          "version": 3
                                          },
                      "file_extension": ".py",
                      "mimetype": "text/x-python",
                      "name": "python",
                      "nbconvert_exporter": "python",
                      "pygments_lexer": "ipython3",
                      "version": "3.9.0"},
    "nbformat": 4,
    "nbformat_minor": 5
}


def parse(script: str, notebook: str):
    with open(script, 'r') as f:
        #
        # create a new notebook
        #
        nb = nbf.NotebookNode(cells=[],
                              metadata=nbf.NotebookNode(metadata),
                              nbformat=4,
                              nbformat_minor=5)

        #
        # create cells for each block of code in the python file
        #
        code_blocks = []
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith('##'):  # <-- new markdown cell delimeter

                #
                # add the current code block as a code cell
                #
                if code_blocks:
                    cell_source = ''.join(code_blocks).strip()
                    if cell_source:
                        cell = nbf.v4.new_code_cell(source=cell_source)
                        cell['id'] = str(uuid.uuid4())
                        nb['cells'].append(cell)
                    code_blocks = []

                #
                # create the markdown cell
                #
                cell = nbf.v4.new_markdown_cell(source=line[2:])
                cell['id'] = str(uuid.uuid4())
                nb['cells'].append(cell)

            else:
                #
                # code cell
                #
                code_blocks.append(line)

                #
                # iff next line is code, accumulate the code block
                #
                if i + 1 < len(lines) and not lines[i + 1].startswith('##'):
                    continue

                #
                # iff next line is not code, add the current code block as a cell
                #
                cell_source = ''.join(code_blocks).strip()
                if cell_source:
                    cell = nbf.v4.new_code_cell(source=cell_source)
                    cell['id'] = str(uuid.uuid4())
                    nb['cells'].append(cell)
                code_blocks = []

        #
        # set the metadata
        #
        nb['metadata'] = metadata

        #
        # write the notebook to file
        #
        nbf.write(nb, notebook)
