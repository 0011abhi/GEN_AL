import fitz

pdf = fitz.open("/Users/ananthas/Desktop/ipc.pdf")
ipc_text = ""
for page in pdf:
    ipc_text += page.get_text()

lines = ipc_text.split("\n")

print("Loading IPC document...")
while True:
    query = input("Ask a question about the IPC: ")
    if query.lower() == "exit":
        print("Goodbye!")
        break
    results = [line for line in lines if query.lower() in line.lower()]
    if results:
        print("\n".join(results[:15]))
    else:
        print("No relevant section found.")
    print("-" * 50)
