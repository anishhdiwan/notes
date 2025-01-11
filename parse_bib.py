import re
import bibtexparser
import os
from lxml import etree

# Helper function to parse a .bib file and return a dictionary of citations
def parse_bib_file(bib_file):
    with open(bib_file, 'r') as f:
        bib_content = f.read()
    bib_data = bibtexparser.loads(bib_content)
    return bib_data.entries  # Returning the list of BibTeX entries

# Helper function to extract last names from author strings
def extract_last_names(authors):
    authors_list = authors.split(" and ")
    return [author.split()[0] for author in authors_list]

# Helper function to generate inline citations and replace in markdown content
def replace_citations(markdown_content, bib_entries, citation_style='apa'):
    # Load CSL style file
    style_file = f'{citation_style}.csl'
    if not os.path.exists(style_file):
        raise FileNotFoundError(f"The CSL file for {citation_style} could not be found.")
    
    # Read the CSL file as bytes
    with open(style_file, 'rb') as f:
        style_content = f.read()

    # Parse the CSL file using lxml (now passing bytes)
    tree = etree.fromstring(style_content)
    
    # Regular expression to match in-line citations (e.g., [@author2020])
    citation_pattern = r"\[@([\w\d, ]+)\]"

    def citation_replacer(match):
        citation_keys = match.group(1).split(",")  # Split on commas for multiple citations
        formatted_citations = []
        
        # Iterate over each citation key
        for citation_key in citation_keys:
            citation_key = citation_key.strip()  # Remove extra spaces
            entry = next((entry for entry in bib_entries if entry['ID'] == citation_key), None)
            if entry:
                # Extract authors and year
                author = entry.get('author', 'Unknown Author')
                year = entry.get('year', 'Unknown Year')

                # Extract last names of authors
                last_names = extract_last_names(author)

                if len(last_names) > 2:
                    # If more than 2 authors, use 'et al.'
                    formatted_author = f"{last_names[0]} et al."
                elif len(last_names) == 2:
                    # If 2 authors, join with 'and'
                    formatted_author = f"{last_names[0]} and {last_names[1]}"
                else:
                    # Single author
                    formatted_author = last_names[0]
                
                # Combine formatted author and year in (Author et al. Year) format
                formatted_citations.append(f"{formatted_author} ({year})")
            else:
                formatted_citations.append(f"[{citation_key}]")  # Fallback if citation is not found
        
        return f"[{', '.join(formatted_citations)}]"

    # Replace all citations in markdown content
    return re.sub(citation_pattern, citation_replacer, markdown_content)

# Helper function to generate the references section
def generate_references(bib_entries, citation_style='apa'):
    # Load CSL style file
    style_file = f'{citation_style}.csl'
    if not os.path.exists(style_file):
        raise FileNotFoundError(f"The CSL file for {citation_style} could not be found.")
    
    # Read the CSL file as bytes
    with open(style_file, 'rb') as f:
        style_content = f.read()

    # Parse the CSL file using lxml (now passing bytes)
    tree = etree.fromstring(style_content)

    references = []
    for entry in bib_entries:
        author = entry.get('author', 'Unknown Author')
        title = entry.get('title', 'Unknown Title')
        year = entry.get('year', 'Unknown Year')
        journal = entry.get('journal', 'Unknown Journal')

        # Manually format references (adjust for your style)
        formatted_reference = f"{author} ({year}). {title}. {journal if journal != 'Unknown Journal' else ''}"
        references.append(formatted_reference)

    return "\n\n#### References\n\n" + "\n\n".join(references)  # Added newline between references

# Main function to process the markdown and BibTeX files
def process_markdown_with_bib(markdown_file, bib_file, citation_style='apa', output_file='output.md'):
    # Parse the BibTeX file
    bib_entries = parse_bib_file(bib_file)
    
    # Read the markdown file
    with open(markdown_file, 'r') as f:
        markdown_content = f.read()
    
    # Replace citations in markdown content
    markdown_content = replace_citations(markdown_content, bib_entries, citation_style)
    
    # Generate the references section
    references = generate_references(bib_entries, citation_style)
    
    # Append the references to the markdown content
    markdown_content += references
    
    # Save the modified markdown content to the output file
    with open(output_file, 'w') as f:
        f.write(markdown_content)

    print(f"Processed markdown saved to {output_file}")


# Example usage
file_name = '2025-01-10-RL-notes-policy_gradients.md'
markdown_file = os.path.join('_posts', file_name)  # Input markdown file
bib_file_name = os.path.splitext(file_name)[0] + '.bib'   # BibTeX file with citation entries
bib_file = os.path.join('_posts/_bibliography', bib_file_name)
citation_style = 'apa'        # Citation style (make sure APA CSL file is available)
# output_file_name = os.path.splitext(file_name)[0] + '_cited' + os.path.splitext(file_name)[-1]  # Output file with citations and references
output_file_name = file_name
output_file = os.path.join('_posts', output_file_name)

process_markdown_with_bib(markdown_file, bib_file, citation_style, output_file)
