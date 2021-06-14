#include <iostream>
#include <vector>
#include <sstream>

using namespace std;

vector<string> split(string str, char delimiter);

int main(){
    string test = "/a/b/c";
    vector<string> result = split(test, '/');
    for (int i=0;i<result.size();i++){
        cout << result[i] << "\n";
    }
}

vector<string> split(string input, char delimiter) {
    vector<string> answer;
    stringstream ss(input);
    string temp;

    while (getline(ss, temp, delimiter)) {
        if( temp.size() > 0)
            answer.push_back(temp);
    }

    return answer;
}