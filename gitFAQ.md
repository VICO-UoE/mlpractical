## How do I get started?

Setting up your virtual environment and getting the initial MLP materials is explained in the first part of the first lab, in `00_Introduction.ipynb`

## How do I update to this weeks coursework?

To avoid potential conflicts between the changes you have made since last week and our additions, we recommend stash your changes and pull the new code from the mlpractical repository by typing:

```
git stash save "Lab1 work"
git pull
```

Then, if you need to, you can always (temporaily) restore a desired state of the repository.


At any point you can use `git stash list` to print a list of all the current stashes. For example if you have one stash created as above the output would be:

```
stash@{0}: On master: Lab 1 work
```

The `stash@{0}` indicates the position of the stash in the stash stack (last in, first out so newer stashes will have lower indices) with 0 meaning this is the entry at the top of stack (and the only entry here). After that is an indication of which branch the stash was made from (here it was made from the default `master` branch), and finally the description you gave the stash when creating it.

To restore changes saved in a stash you have two options. 

The easiest option is to create a new *branch* in your local repository from the stash. A *branch* can be thought of as a parallel working copy of the repository, which is 'branched' off from a particular commit version of another branch, often as here the main 'master' branch. 

First you need to make sure any changes made to the current branch are either committed or stashed using `git stash save` as above. You can check if you have any non staged changes since the last commit by running `git status` - if there are any they will be listed under a `Changes not staged for commit` section of the output.

Once all changes are committed / stashed, you can then create a new branch from your previous stash. First run `git stash list` as above and take note of the stash index `i` in the `stash@{i}` indicator corresponding to the stash you wish to restore.

Then run

```
git stash branch name_of_branch stash@{i}
```
where `name_of_branch` is some name to give the branch (e.g. `lab1`). This will take your stashed changes and apply them to the commit they were derived from in a new branch. If you run `git branch` you should now see something like

```
* lab1
  master
```
where the asterisk indicates the currently active branch. Your current working copy should now be in an identical state to the point when you made the stash. You can now continue working from this branch and changes you make and commit will be on this branch alone.

If you later want to return to the master branch (for example to pull some changes from github at the start of a new lab) you can do this by running

```
git checkout master
```

Again you need to make sure there are not any uncommitted stages on your current branch before you do this. Similarly you can use `git checkout branch_name` to check out any branch.

The alternative to creating a separate branch to restore a stash to is to use `git stash apply` or `git stash pop` to merge the stashed changes in to your current branch working copy (apply keeps the stash after applying it while pop removes it from the stash stack). If the stashed changes involve updates to files you have also edited in your current working copy you will most probably end up with merge conflicts you will need to resolve. If you are comfortable doing this feel free to use this approach, however this is not something we will be able to help you with.


## If I find an error in something in the practical, can a I push a change to correct it?

Yes by making a fork, then a pull request.  But equally you can make a comment on nb.mit.edu or send email.  Again probably not worth bothering with the git way unless you already understand what to do.

## What is a good tutorial on git?

I like [this concise one from Roger Dudler](http://rogerdudler.github.io/git-guide/)  and a [slightly longer one from Atlassian](https://www.atlassian.com/git/tutorials/).  There are many others!
