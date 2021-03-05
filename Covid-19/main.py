from flask import Flask ,render_template,request
app = Flask(__name__)
import pickle

file=open('model.pkl','rb')
clf=pickle.load(file)
file.close()

@app.route('/', methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        bodypain = int(myDict['bodypain'])
        runnynose = int(myDict['runnynose'])
        diffBreath = int(myDict['diffBreath'])
        # Code for inference
        inputFeatures=[fever,age,bodypain,runnynose,diffBreath]
        infprob=clf.predict([inputFeatures])[0]
        if infprob < 0:
            infprob=infprob * (-1)
        print(infprob)
        return render_template('show.html',inf=round(infprob*100))
    return render_template('index.html')
        # return 'Hello, World!' + str(infprob)


if __name__ == "__main__":
    app.run(debug=True)