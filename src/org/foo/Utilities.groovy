package org.foo
class Utilities implements Serializable {
	def steps

	Utilities(){
	}

	Utilities(steps) {
		this.steps = steps
	}
	def mvn(args) {
		return this.steps.env.HOME
	}
	
	static def mvn(step, args) {
		return step.env.BRANCH_NAME
	}

}
